# ============================================================================
# FALSE-STABILITY - Fase 3: lineas base M0, M1, M2
#   M0  NEWS2 modificado del ultimo set (practica habitual)
#   M1  "Foto final": estaticas + historia + ultimo set
#   M2  M1 + resumen de trayectoria + controles + tratamiento en urgencia
# Particion temporal por anchor_year_group (separa pacientes por construccion).
# Todo lo que imprime y los graficos son agregados y se pueden compartir.
# ============================================================================
import os, zipfile, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
FOLDER = globals().get("FOLDER", "/content/drive/MyDrive/ARTICULOS MIOS/False Stability")
OUT = globals().get("OUT", FOLDER + "/salida")
ED_ZIP = os.path.join(FOLDER, "mimic-iv-ed-2.2.zip")
N_BOOT = 200
SEED = 42
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def load_out(n):
    p = os.path.join(OUT, n + ".parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.read_csv(os.path.join(OUT, n + ".csv.gz"))


def read_from_zip(table, **kw):
    with zipfile.ZipFile(ED_ZIP) as z:
        names = [n for n in z.namelist() if os.path.basename(n).startswith(table + ".csv")]
        if not names:
            return None
        with z.open(names[0]) as f:
            return pd.read_csv(f, compression="gzip" if names[0].endswith(".gz") else None, **kw)


print("Leyendo tablas auxiliares...")
ed_all = read_from_zip("edstays", usecols=["subject_id", "stay_id", "intime", "outtime"],
                       parse_dates=["intime", "outtime"])
pyxis = read_from_zip("pyxis", usecols=["stay_id", "charttime", "name"], parse_dates=["charttime"])
adm_all = pd.read_csv(os.path.join(FOLDER, "admissions.csv.gz"),
                      usecols=["subject_id", "hadm_id", "admittime"], parse_dates=["admittime"])
pats = pd.read_csv(os.path.join(FOLDER, "patients.csv.gz"), usecols=["subject_id", "anchor_year_group"])


def history_features(ed_all, adm_all):
    """Visitas a urgencia y hospitalizaciones previas (ventanas 180 y 365 dias, excluye la actual)."""
    e = ed_all[["subject_id", "stay_id", "intime"]].rename(columns={"intime": "t"})
    e["is_ed"], e["is_adm"] = 1, 0
    a = adm_all[["subject_id", "admittime"]].rename(columns={"admittime": "t"})
    a["stay_id"], a["is_ed"], a["is_adm"] = -1, 0, 1
    ev = pd.concat([e, a], ignore_index=True).sort_values(["subject_id", "t"]).reset_index(drop=True)
    out = ev[["stay_id"]].copy()
    for d in (180, 365):
        r = (ev.set_index("t").groupby("subject_id")[["is_ed", "is_adm"]]
             .rolling(f"{d}D", closed="left").sum().reset_index(drop=True))
        out[f"prev_ed_{d}d"] = r["is_ed"].fillna(0).values
        out[f"prev_adm_{d}d"] = r["is_adm"].fillna(0).values
    return out[out["stay_id"] >= 0]


MED_GROUPS = {
    "tx_fluidos": ["sodium chloride", "lactated ringer", "normal saline"],
    "tx_antibiotico": ["ceftriaxone", "cefazolin", "cephalexin", "cefepime", "vancomycin", "piperacillin",
                       "zosyn", "levofloxacin", "ciprofloxacin", "azithromycin", "doxycycline",
                       "metronidazole", "amoxicillin", "augmentin", "clindamycin", "nitrofurantoin",
                       "sulfamethoxazole", "bactrim", "meropenem", "ampicillin", "gentamicin"],
    "tx_antipiretico": ["acetaminophen", "ibuprofen", "ketorolac"],
    "tx_opioide": ["morphine", "hydromorphone", "fentanyl", "oxycodone", "percocet", "tramadol"],
    "tx_antiemetico": ["ondansetron", "metoclopramide", "prochlorperazine", "promethazine"],
    "tx_broncodilatador": ["albuterol", "ipratropium", "duoneb"],
    "tx_benzodiacepina": ["lorazepam", "diazepam", "midazolam"],
    "tx_diuretico": ["furosemide"],
}


def treatment_features(pyxis, ed_all):
    cols = ["stay_id"] + list(MED_GROUPS) + ["tx_n_meds"]
    if pyxis is None:
        print("  (pyxis no encontrado: se omiten variables de tratamiento)")
        return pd.DataFrame(columns=cols)
    p = pyxis.merge(ed_all[["stay_id", "outtime"]], on="stay_id")
    p = p[p["charttime"] <= p["outtime"]]
    nm = p["name"].astype(str).str.lower()
    flush = nm.str.contains("flush")
    for g, kws in MED_GROUPS.items():
        p[g] = nm.str.contains("|".join(kws), regex=True)
        if g == "tx_fluidos":
            p[g] &= ~flush
    agg = p.groupby("stay_id")[list(MED_GROUPS)].any().astype(int)
    agg["tx_n_meds"] = p.groupby("stay_id")["name"].nunique()
    return agg.reset_index()


hist = history_features(ed_all, adm_all)
tx = treatment_features(pyxis, ed_all)

# ----------------------------------------------------------------------------
# Variables por modelo
# ----------------------------------------------------------------------------
CAT = ["sexo", "esi", "llegada", "motivo"]
STATIC_NUM = ["age", "pain_num", "prev_ed_180d", "prev_ed_365d", "prev_adm_180d", "prev_adm_365d"]
LAST_NUM = ["heartrate_last", "resprate_last", "o2sat_last", "sbp_last", "temperature_last",
            "news2_last", "maxcomp_last", "shock_index_last"]
TRAJ_NUM = ["news2_first", "news2_max", "news2_prior_max", "news2_drop", "n_sets", "max_gap_h",
            "hours_last_set_to_exit", "ed_los_h", "sets_per_hour"] + \
           [f"{v}_{s}" for v in ["heartrate", "resprate", "o2sat", "sbp", "temperature"] for s in ("max", "min")] + \
           ["apparent_normalization_i", "prior_abnormal_i"]
TX_NUM = list(MED_GROUPS) + ["tx_n_meds"]

FEATS = {
    "M1": dict(num=STATIC_NUM + LAST_NUM, cat=CAT),
    "M2": dict(num=STATIC_NUM + LAST_NUM + TRAJ_NUM + TX_NUM, cat=CAT),
}


def prep(c, top_cc=50):
    c = c.merge(hist, on="stay_id", how="left").merge(tx, on="stay_id", how="left")
    c = c.merge(pats, on="subject_id", how="left")
    for col in TX_NUM:
        c[col] = c[col].fillna(0)
    c["y"] = c["outcome"].astype(int)
    c["sexo"] = c["gender"].fillna("NA").astype(str)
    c["esi"] = c["acuity"].fillna(-1).astype(int).astype(str).replace("-1", "NA")
    c["llegada"] = c["arrival_transport"].fillna("NA").astype(str)
    cc = c["chiefcomplaint"].fillna("NA").astype(str).str.lower().str.strip()
    c["motivo"] = np.where(cc.isin(cc.value_counts().index[:top_cc]), cc, "otro")
    c["pain_num"] = pd.to_numeric(c["pain"], errors="coerce").where(lambda s: s.between(0, 10))
    c["apparent_normalization_i"] = c["apparent_normalization"].astype(int)
    c["prior_abnormal_i"] = c["prior_abnormal"].astype(int)
    start = c["anchor_year_group"].astype(str).str[:4]
    c["split"] = np.where(pd.to_numeric(start, errors="coerce") >= 2017, "test", "train")
    return c


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
def fit_m0(tr):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(tr[["news2_last"]].fillna(tr["news2_last"].median()), tr["y"])
    return lambda d: lr.predict_proba(d[["news2_last"]].fillna(tr["news2_last"].median()))[:, 1]


def fit_lr(tr, spec):
    ct = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), spec["num"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), spec["cat"]),
    ])
    m = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000))
    m.fit(tr[spec["num"] + spec["cat"]], tr["y"])
    return lambda d: m.predict_proba(d[spec["num"] + spec["cat"]])[:, 1]


def fit_gbm(tr, spec):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    Xc = enc.fit_transform(tr[spec["cat"]].astype(str))
    X = np.hstack([tr[spec["num"]].to_numpy(float), Xc])
    mask = [False] * len(spec["num"]) + [True] * len(spec["cat"])
    m = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, categorical_features=mask, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=50, scoring="loss", random_state=SEED)
    m.fit(X, tr["y"])

    def pred(d):
        Xd = np.hstack([d[spec["num"]].to_numpy(float), enc.transform(d[spec["cat"]].astype(str))])
        return m.predict_proba(Xd)[:, 1]
    return pred


# ----------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------
def cal_slope(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return lr.coef_[0, 0]


def metrics(y, p):
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "O/E": y.mean() / p.mean(), "pendiente": cal_slope(y, p)}


def boot_ci(y, preds, ref=None):
    """IC95 por bootstrap; si ref, tambien diferencia de AUPRC vs ref (pareada)."""
    n = len(y)
    res = {k: [] for k in preds}
    diffs = {k: [] for k in preds}
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        ap = {k: average_precision_score(y[i], p[i]) for k, p in preds.items()}
        au = {k: roc_auc_score(y[i], p[i]) for k, p in preds.items()}
        for k in preds:
            res[k].append((au[k], ap[k]))
            if ref:
                diffs[k].append(ap[k] - ap[ref])
    out = {}
    for k in preds:
        a = np.array(res[k])
        out[k] = dict(AUROC_ic=f"{np.percentile(a[:, 0], 2.5):.3f}-{np.percentile(a[:, 0], 97.5):.3f}",
                      AUPRC_ic=f"{np.percentile(a[:, 1], 2.5):.3f}-{np.percentile(a[:, 1], 97.5):.3f}")
        if ref and k != ref:
            d = np.array(diffs[k])
            out[k]["dAUPRC_vs_" + ref] = f"{np.mean(d):+.3f} ({np.percentile(d, 2.5):+.3f} a {np.percentile(d, 97.5):+.3f})"
    return out


def evaluate(te, preds, nombre, subset):
    y = te["y"].to_numpy()
    ci = boot_ci(y, {k: p for k, p in preds.items()}, ref="M1_gbm")
    rows = []
    for k, p in preds.items():
        m = metrics(y, p)
        rows.append({"cohorte": nombre, "subgrupo": subset, "modelo": k, "n": len(y), "eventos": int(y.sum()),
                     "AUROC": round(m["AUROC"], 3), "AUROC_IC95": ci[k]["AUROC_ic"],
                     "AUPRC": round(m["AUPRC"], 3), "AUPRC_IC95": ci[k]["AUPRC_ic"],
                     "Brier": round(m["Brier"], 4), "O/E": round(m["O/E"], 2),
                     "pendiente_cal": round(m["pendiente"], 2),
                     "dAUPRC_vs_M1_gbm": ci[k].get("dAUPRC_vs_M1_gbm", "")})
    return rows


def discordance(te, p_m2, thr, nombre):
    """Pacientes con NEWS2 final normal (<=1): riesgo M2 alto vs bajo."""
    d = te.assign(p=p_m2)
    d = d[d["news2_last"] <= 1]
    d["riesgo_M2"] = np.where(d["p"] >= thr, "alto (top 10% train)", "bajo")
    t = d.groupby("riesgo_M2")["y"].agg(n="size", eventos="sum", tasa_pct=lambda s: round(100 * s.mean(), 2))
    t = t.reset_index()
    t.insert(0, "cohorte", nombre)
    t["pct_de_NEWS2_normal"] = (100 * t["n"] / t["n"].sum()).round(1)
    return t


def decision_curve(te, preds, thresholds, nombre):
    y = te["y"].to_numpy()
    n, prev = len(y), y.mean()
    rows = []
    for pt in thresholds:
        r = {"cohorte": nombre, "umbral_pct": round(100 * pt, 1),
             "tratar_todos": round(prev - (1 - prev) * pt / (1 - pt), 4)}
        for k, p in preds.items():
            tp = ((p >= pt) & (y == 1)).sum() / n
            fp = ((p >= pt) & (y == 0)).sum() / n
            r[k] = round(tp - fp * pt / (1 - pt), 4)
        rows.append(r)
    return pd.DataFrame(rows)


def calibration_plot(te, preds, nombre, ax):
    y = te["y"].to_numpy()
    for k, p in preds.items():
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
        ax.plot(100 * g["p"], 100 * g["y"], marker="o", label=k)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1)
    ax.set(title=nombre, xlabel="Riesgo predicho (%)", ylabel="Tasa observada (%)")
    ax.legend(fontsize=8)


# ----------------------------------------------------------------------------
# Ejecucion
# ----------------------------------------------------------------------------
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
all_rows, disc_all, dca_all = [], [], []
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
DCA_THR = {"A: reconsulta 72h": [0.01, 0.02, 0.03, 0.05, 0.08],
           "B: UCI 48h": [0.02, 0.04, 0.06, 0.10, 0.15]}

for (fname, nombre), ax in zip((("cohorte_A", "A: reconsulta 72h"), ("cohorte_B", "B: UCI 48h")), axes):
    print(f"\n### {nombre}")
    c = prep(load_out(fname))
    tr, te = c[c["split"] == "train"], c[c["split"] == "test"]
    print(f"  train n={len(tr):,} eventos={tr['y'].sum():,} | test n={len(te):,} eventos={te['y'].sum():,}")

    models = {"M0_news2": fit_m0(tr),
              "M1_lr": fit_lr(tr, FEATS["M1"]), "M1_gbm": fit_gbm(tr, FEATS["M1"]),
              "M2_lr": fit_lr(tr, FEATS["M2"]), "M2_gbm": fit_gbm(tr, FEATS["M2"])}
    print("  modelos entrenados")
    preds = {k: f(te) for k, f in models.items()}

    all_rows += evaluate(te, preds, nombre, "toda la cohorte de prueba")
    an = te["apparent_normalization"].to_numpy()
    all_rows += evaluate(te[an], {k: p[an] for k, p in preds.items()}, nombre, "normalizacion aparente")

    thr = np.quantile(models["M2_gbm"](tr), 0.90)
    disc_all.append(discordance(te, preds["M2_gbm"], thr, nombre))
    dca_all.append(decision_curve(te, {k: preds[k] for k in ("M0_news2", "M1_gbm", "M2_gbm")},
                                  DCA_THR[nombre], nombre))
    calibration_plot(te, {k: preds[k] for k in ("M0_news2", "M1_gbm", "M2_gbm")}, nombre, ax)

res = pd.DataFrame(all_rows)
disc = pd.concat(disc_all)
dca = pd.concat(dca_all)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fase3_calibracion.png"), dpi=150)
plt.show()

for titulo, df, f in (("Desempeno en prueba temporal (IC95 bootstrap)", res, "fase3_desempeno"),
                      ("Discordancia: NEWS2 final normal, riesgo M2", disc, "fase3_discordancia"),
                      ("Beneficio neto (curva de decision)", dca, "fase3_dca")):
    print(f"\n=== {titulo} ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, f + ".csv"), index=False)
