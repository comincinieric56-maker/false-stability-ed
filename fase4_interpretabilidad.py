# ============================================================================
# FALSE-STABILITY - Fase 4: ablaciones e interpretabilidad (SHAP) de M2
# Requiere haber corrido la celda de la fase 3 en esta misma sesion
# (reutiliza prep, fit_gbm, boot_ci, FEATS y los grupos de variables).
# Todo lo que imprime y los graficos son agregados y se pueden compartir.
# ============================================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
for _need in ("prep", "fit_gbm", "boot_ci", "FEATS", "load_out"):
    if _need not in globals():
        raise RuntimeError("Primero corre la celda de la fase 3 en esta sesion (define prep, fit_gbm, etc.).")

try:
    import lightgbm as lgb
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "lightgbm"], check=True)
    import lightgbm as lgb

# ----------------------------------------------------------------------------
# Grupos de variables para ablaciones
# ----------------------------------------------------------------------------
HIST = ["prev_ed_180d", "prev_ed_365d", "prev_adm_180d", "prev_adm_365d"]
STATIC = ["age", "pain_num"]
LAST = LAST_NUM
TRAJ = ["news2_first", "news2_max", "news2_prior_max", "news2_drop",
        "apparent_normalization_i", "prior_abnormal_i"] + \
       [f"{v}_{s}" for v in ["heartrate", "resprate", "o2sat", "sbp", "temperature"] for s in ("max", "min")]
PROCESS = ["n_sets", "max_gap_h", "hours_last_set_to_exit", "ed_los_h", "sets_per_hour"]
TX = TX_NUM
FULL = STATIC + HIST + LAST + TRAJ + PROCESS + TX
GROUP_OF = {**{f: "estaticas" for f in STATIC + CAT}, **{f: "historia" for f in HIST},
            **{f: "ultimo set" for f in LAST}, **{f: "trayectoria" for f in TRAJ},
            **{f: "controles/proceso" for f in PROCESS}, **{f: "tratamiento" for f in TX}}

ABLACIONES = {
    "M2 completo": FULL,
    "- trayectoria": [f for f in FULL if f not in TRAJ],
    "- controles/proceso": [f for f in FULL if f not in PROCESS],
    "- tratamiento": [f for f in FULL if f not in TX],
    "- historia": [f for f in FULL if f not in HIST],
    "- ultimo set": [f for f in FULL if f not in LAST],
    "solo estaticas + trayectoria": STATIC + TRAJ,
}

KEY_DEP = ["news2_prior_max", "news2_drop", "resprate_max", "o2sat_min", "heartrate_max", "sbp_min",
           "temperature_max", "age", "n_sets", "hours_last_set_to_exit", "ed_los_h", "tx_n_meds"]


# ----------------------------------------------------------------------------
# 1. Ablaciones (mismo gradient boosting de la fase 3)
# ----------------------------------------------------------------------------
def ablaciones(tr, te, nombre):
    preds = {}
    for k, num in ABLACIONES.items():
        cat = CAT if k != "solo estaticas + trayectoria" else ["sexo", "esi"]
        preds[k] = fit_gbm(tr, dict(num=num, cat=cat))(te)
        print(f"    {k}")
    rows = []
    an = te["apparent_normalization"].to_numpy()
    for subset, mask in (("toda la prueba", np.ones(len(te), bool)), ("normalizacion aparente", an)):
        y = te["y"].to_numpy()[mask]
        pp = {k: p[mask] for k, p in preds.items()}
        ci = boot_ci(y, pp, ref="M2 completo")
        for k, p in pp.items():
            rows.append({"cohorte": nombre, "subgrupo": subset, "variante": k,
                         "AUROC": round(roc_auc_score(y, p), 3), "AUPRC": round(average_precision_score(y, p), 3),
                         "AUPRC_IC95": ci[k]["AUPRC_ic"],
                         "dAUPRC_vs_completo": ci[k].get("dAUPRC_vs_M2 completo", "")})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. SHAP exacto (TreeSHAP nativo de LightGBM)
# ----------------------------------------------------------------------------
def to_lgb(d, cats_ref=None):
    X = d[FULL + CAT].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c].astype(str), categories=None if cats_ref is None else cats_ref[c])
    return X


def fit_lgb(tr):
    X = to_lgb(tr)
    cats = {c: X[c].cat.categories for c in CAT}
    ntr = int(0.85 * len(X))
    idx = np.random.default_rng(SEED).permutation(len(X))
    i_fit, i_val = idx[:ntr], idx[ntr:]
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=SEED, verbose=-1)
    m.fit(X.iloc[i_fit], tr["y"].iloc[i_fit], eval_set=[(X.iloc[i_val], tr["y"].iloc[i_val])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, cats


def shap_tables(m, cats, te, nombre):
    X = to_lgb(te, cats)
    p = m.predict_proba(X)[:, 1]
    sv = pd.DataFrame(m.predict(X, pred_contrib=True)[:, :-1], columns=X.columns, index=te.index)
    an = te["apparent_normalization"].to_numpy()
    y = te["y"].to_numpy()

    imp = pd.DataFrame({
        "media_|SHAP|_todos": sv.abs().mean(),
        "media_|SHAP|_normalizacion": sv[an].abs().mean(),
        "SHAP_medio_eventos_normalizacion": sv[an & (y == 1)].mean(),
        "SHAP_medio_no_eventos_normalizacion": sv[an & (y == 0)].mean(),
    })
    imp["grupo"] = imp.index.map(GROUP_OF)
    imp = imp.sort_values("media_|SHAP|_normalizacion", ascending=False).round(4)
    imp.insert(0, "cohorte", nombre)

    grp = imp.groupby("grupo")[["media_|SHAP|_todos", "media_|SHAP|_normalizacion"]].sum()
    grp = (100 * grp / grp.sum()).round(1).sort_values("media_|SHAP|_normalizacion", ascending=False)
    grp.columns = ["pct_importancia_todos", "pct_importancia_normalizacion"]
    grp.insert(0, "cohorte", nombre)

    # Dependencia: quintiles de cada variable clave dentro de la normalizacion aparente
    dep = []
    d = te[an].assign(_y=y[an])
    for f in KEY_DEP:
        if f not in d or d[f].nunique() < 3:
            continue
        q = pd.qcut(d[f], 5, duplicates="drop")
        g = pd.DataFrame({"tramo": q.astype(str), "shap": sv.loc[d.index, f], "y": d["_y"]}).groupby("tramo", observed=True)
        t = g.agg(n=("y", "size"), tasa_pct=("y", lambda s: round(100 * s.mean(), 2)),
                  SHAP_medio=("shap", lambda s: round(s.mean(), 3))).reset_index()
        t.insert(0, "variable", f)
        t.insert(0, "cohorte", nombre)
        dep.append(t)
    dep = pd.concat(dep) if dep else pd.DataFrame()

    # Fenotipo: normalizacion aparente con riesgo alto vs bajo (top 20% del subgrupo)
    pa = p[an]
    hi = pa >= np.quantile(pa, 0.80)
    sub = te[an]
    cols = ["age", "news2_first", "news2_prior_max", "news2_drop", "heartrate_max", "resprate_max",
            "o2sat_min", "sbp_min", "temperature_max", "n_sets", "ed_los_h", "hours_last_set_to_exit",
            "prev_adm_365d", "tx_antibiotico", "tx_fluidos", "tx_antipiretico", "tx_n_meds"]
    fen = pd.DataFrame({
        "riesgo alto (top 20%)": sub[hi][cols].median(),
        "riesgo bajo": sub[~hi][cols].median(),
    })
    for c in ("tx_antibiotico", "tx_fluidos", "tx_antipiretico"):
        fen.loc[c] = [round(100 * sub[hi][c].mean(), 1), round(100 * sub[~hi][c].mean(), 1)]
    fen = fen.rename(index={c: c + " (%)" for c in ("tx_antibiotico", "tx_fluidos", "tx_antipiretico")})
    fen.loc["n"] = [int(hi.sum()), int((~hi).sum())]
    fen.loc["tasa desenlace (%)"] = [round(100 * y[an][hi].mean(), 2), round(100 * y[an][~hi].mean(), 2)]
    fen = fen.round(2).reset_index().rename(columns={"index": "variable (mediana)"})
    fen.insert(0, "cohorte", nombre)

    return imp, grp, dep, fen, sv[an], round(average_precision_score(y, p), 3)


# ----------------------------------------------------------------------------
# Ejecucion
# ----------------------------------------------------------------------------
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
abl_all, imp_all, grp_all, dep_all, fen_all = [], [], [], [], []
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for (fname, nombre), ax in zip((("cohorte_B", "B: UCI 48h"), ("cohorte_A", "A: reconsulta 72h")), axes):
    print(f"\n### {nombre}")
    c = prep(load_out(fname))
    tr, te = c[c["split"] == "train"], c[c["split"] == "test"].copy()
    print("  ablaciones:")
    abl_all.append(ablaciones(tr, te, nombre))
    print("  SHAP...")
    m, cats = fit_lgb(tr)
    imp, grp, dep, fen, sv_an, ap = shap_tables(m, cats, te, nombre)
    print(f"  AUPRC LightGBM (control con fase 3): {ap}")
    imp_all.append(imp); grp_all.append(grp); dep_all.append(dep); fen_all.append(fen)

    top = imp.head(15).iloc[::-1]
    colors = top["grupo"].map({"trayectoria": "#2a7", "controles/proceso": "#e80", "tratamiento": "#a5c",
                               "ultimo set": "#37c", "historia": "#888", "estaticas": "#bbb"})
    ax.barh(top.index, top["media_|SHAP|_normalizacion"], color=colors)
    ax.set(title=f"{nombre}: normalizacion aparente", xlabel="media |SHAP| (log-odds)")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#2a7", "#e80", "#a5c", "#37c", "#888", "#bbb")]
fig.legend(handles, ["trayectoria", "controles/proceso", "tratamiento", "ultimo set", "historia", "estaticas"],
           loc="lower center", ncol=6, frameon=False)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(os.path.join(OUT, "fase4_shap.png"), dpi=150)
plt.show()

for titulo, dfs, f in (("Ablaciones (gradient boosting, IC95 bootstrap)", abl_all, "fase4_ablaciones"),
                       ("Importancia por grupo de variables (% del total |SHAP|)", grp_all, "fase4_grupos"),
                       ("Top 20 variables en normalizacion aparente", [i.head(20).reset_index().rename(columns={"index": "variable"}) for i in imp_all], "fase4_top20"),
                       ("Fenotipo de falsa estabilidad: riesgo alto vs bajo", fen_all, "fase4_fenotipo"),
                       ("Dependencia por quintiles (normalizacion aparente)", dep_all, "fase4_dependencia")):
    df = pd.concat(dfs)
    print(f"\n=== {titulo} ===")
    print(df.to_string(index=("grupos" in f)))
    df.to_csv(os.path.join(OUT, f + ".csv"), index=("grupos" in f))
