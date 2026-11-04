# ============================================================================
# FALSE-STABILITY - Fase 7c (parte 1): sensibilidad N4, verificacion de la
# calibracion temporal y Tabla 1 recalculada. Ultimo analisis antes de cerrar.
#
# Modelos (regresion logistica, mismos predictores y muestra para todos):
#   N2   modelo clinico basal (edad, sexo, ESI, motivo, llegada) + NEWS2 final
#   N3   N2 + NEWS2 maximo previo
#   N2P  N2 + proceso de atencion (log n_sets, estadia, horas desde el ultimo set)
#   N4   N3 + proceso de atencion                 <- sensibilidad preespecificada
# Contrastes: N3-N2 (replica de 7b en la muestra comun), N4-N2P (aporte de la
# gravedad previa neto de la intensidad de vigilancia), N4-N3.
#
# Calibracion: pendiente y calibration-in-the-large (CITL) con IC95% por
# bootstrap agrupado por paciente, en toda la cohorte y en el subgrupo con
# normalizacion aparente, tanto en VC agrupada como en la particion temporal.
# "CITL relativo" = CITL del subgrupo - CITL de toda la cohorte: separa la
# subestimacion propia del subgrupo de la deriva temporal de la prevalencia.
# Curvas por quintil/decil (solo conteos agregados) para la Figura 4.
#
# Semilla 20261103, 1.000 replicas. CPU, ~35-45 min. Salida solo agregada.
# ============================================================================
import os, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

FOLDER = globals().get("FOLDER", "/content/drive/MyDrive/ARTICULOS MIOS/False Stability")
S_MI, S_MC, MC = (os.path.join(FOLDER, x) for x in ("salida", "salida_mcmed", "mcmed"))
OUT7 = os.path.join(FOLDER, "salida_fase7")
os.makedirs(OUT7, exist_ok=True)

SEED, N_BOOT, K_FOLDS = 20261103, 1000, 5
BASE_CAT = ["sexo", "esi", "motivo", "llegada"]
PROC = ["log_sets", "los_h", "h_ult"]
NUM = {"N2": ["age", "news2_last"], "N3": ["age", "news2_last", "news2_prior_max"],
       "N2P": ["age", "news2_last"] + PROC, "N4": ["age", "news2_last", "news2_prior_max"] + PROC}
CONTRASTES = [("N3", "N2"), ("N4", "N2P"), ("N4", "N3")]


def load_any(folder, name):
    for ext, rd in ((".parquet", pd.read_parquet), (".csv.gz", lambda p: pd.read_csv(p, low_memory=False))):
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return rd(p)
    raise FileNotFoundError(f"{name} no esta en {folder}")


def prep(c, top_cc=30):
    """Identica a 7a/7b (definiciones congeladas)."""
    c = c.copy()
    c["y"] = c["outcome"].astype(bool).astype(int)
    c["an"] = c["apparent_normalization"].astype(bool)
    c["grupo"] = np.select([c["an"], c["stable_throughout"].astype(bool)], ["normalizacion", "estable"], "ultimo_anormal")
    c["sexo"] = c["gender"].where(c["gender"].isin(["F", "M"]))
    e = pd.to_numeric(c["acuity"], errors="coerce").round()
    c["esi"] = e.astype("Int64").astype(str).replace("<NA>", np.nan)
    arr = c["arrival_transport"].fillna("").astype(str).str.upper()
    c["llegada"] = np.select([arr.str.contains("WALK|SELF"), arr.str.contains("AMBUL|EMS|HELI")], ["propio", "ems"], "otro")
    cc = c["chiefcomplaint"].fillna("NA").astype(str).str.lower().str.strip()
    c["motivo"] = np.where(cc.isin(cc.value_counts().index[:top_cc]), cc, "otro")
    c["age"] = pd.to_numeric(c["age"], errors="coerce")
    c["news2_prior_max"] = pd.to_numeric(c["news2_prior_max"], errors="coerce")
    c["log_sets"] = np.log(c["n_sets"])
    c["los_h"] = pd.to_numeric(c["ed_los_h"], errors="coerce").clip(0, 72)
    c["h_ult"] = pd.to_numeric(c["hours_last_set_to_exit"], errors="coerce").clip(0, 48)
    return c


def model(name):
    ct = ColumnTransformer([("num", StandardScaler(), NUM[name]),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), BASE_CAT)])
    return make_pipeline(ct, LogisticRegression(C=1e4, max_iter=3000))


def oof_predictions(d):
    P = {m: np.full(len(d), np.nan) for m in NUM}
    for tr, te in GroupKFold(n_splits=K_FOLDS).split(d, d["y"], d["subject_id"]):
        for m in NUM:
            f = model(m).fit(d.iloc[tr][NUM[m] + BASE_CAT], d["y"].iloc[tr])
            P[m][te] = f.predict_proba(d.iloc[te][NUM[m] + BASE_CAT])[:, 1]
    return P


def holdout_predictions(tr, te):
    return {m: model(m).fit(tr[NUM[m] + BASE_CAT], tr["y"]).predict_proba(te[NUM[m] + BASE_CAT])[:, 1] for m in NUM}


# ----------------------------------------------------------------------------
# Calibracion rapida (Newton) para poder hacer bootstrap
# ----------------------------------------------------------------------------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def citl_fast(y, lp):
    a = 0.0
    for _ in range(25):
        p = 1 / (1 + np.exp(-(lp + a)))
        g, h = (y - p).sum(), (p * (1 - p)).sum()
        step = g / max(h, 1e-12)
        a += step
        if abs(step) < 1e-8:
            break
    return a


def slope_fast(y, lp):
    b = np.array([0.0, 1.0])
    X = np.column_stack([np.ones_like(lp), lp])
    for _ in range(30):
        p = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.nan
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b[1]


def cluster_index_sampler(groups, rng):
    order = np.argsort(groups, kind="stable")
    uniq, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)

    def draw():
        s = rng.integers(0, len(uniq), len(uniq))
        lens = counts[s]
        base = np.repeat(starts[s], lens)
        offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        return order[base + offs]
    return draw


def metrics(y, p, lp):
    if y.sum() < 5 or y.sum() == len(y):
        return (np.nan,) * 4          # <5 eventos: no estimable
    return (roc_auc_score(y, p), average_precision_score(y, p), slope_fast(y, lp), citl_fast(y, lp))


def evaluate(y, P, groups, an, rng, etiqueta):
    """Toda la cohorte y subgrupo AN en las MISMAS replicas (IC conjuntos)."""
    LP = {m: logit(p) for m, p in P.items()}
    draw = cluster_index_sampler(groups, rng)
    point = {(m, s): metrics(y[msk], P[m][msk], LP[m][msk]) for m in P for s, msk in (("toda", slice(None)), ("an", an))}
    boot = {k: [] for k in point}
    for _ in range(N_BOOT):
        i = draw()
        ia = i[an[i]]
        if y[i].sum() == 0:
            continue
        for m in P:
            boot[(m, "toda")].append(metrics(y[i], P[m][i], LP[m][i]))
            boot[(m, "an")].append(metrics(y[ia], P[m][ia], LP[m][ia]))
    B = {k: np.array(v, dtype=float).reshape(-1, 4) for k, v in boot.items()}
    def ci(a, d=3):
        a = a[np.isfinite(a)]
        return f"{np.percentile(a, 2.5):.{d}f} a {np.percentile(a, 97.5):.{d}f}" if len(a) >= 20 else "no estimable"
    perf, diffs = [], []
    for m in P:
        for s, msk, nom in (("toda", slice(None), "toda la cohorte"), ("an", an, "normalizacion aparente")):
            pt, bb = point[(m, s)], B[(m, s)]
            row = {**etiqueta, "subgrupo": nom, "modelo": m, "n": int(len(y[msk])), "eventos": int(y[msk].sum()),
                   "AUROC": round(pt[0], 3), "AUROC_IC95": ci(bb[:, 0]), "AUPRC": round(pt[1], 3), "AUPRC_IC95": ci(bb[:, 1]),
                   "pendiente": round(pt[2], 2), "pendiente_IC95": ci(bb[:, 2], 2),
                   "CITL": round(pt[3], 3), "CITL_IC95": ci(bb[:, 3])}
            if s == "an":
                rel = bb[:, 3] - B[(m, "toda")][:, 3]
                row.update({"CITL_relativo": round(pt[3] - point[(m, "toda")][3], 3), "CITL_relativo_IC95": ci(rel),
                            "OR_subestimacion": round(float(np.exp(pt[3] - point[(m, "toda")][3])), 2)})
            perf.append(row)
    for a, b in CONTRASTES:
        for s, nom in (("toda", "toda la cohorte"), ("an", "normalizacion aparente")):
            da = B[(a, s)][:, 0] - B[(b, s)][:, 0]
            dp = B[(a, s)][:, 1] - B[(b, s)][:, 1]
            dc = B[(a, s)][:, 3] - B[(b, s)][:, 3]
            diffs.append({**etiqueta, "subgrupo": nom, "contraste": f"{a} - {b}",
                          "dAUROC": round(point[(a, s)][0] - point[(b, s)][0], 4), "dAUROC_IC95": ci(da, 4),
                          "p_bootstrap_dAUROC<=0": round(float((da[np.isfinite(da)] <= 0).mean()), 4) if np.isfinite(da).sum() >= 20 else np.nan,
                          "dAUPRC": round(point[(a, s)][1] - point[(b, s)][1], 4), "dAUPRC_IC95": ci(dp, 4),
                          "dCITL": round(point[(a, s)][3] - point[(b, s)][3], 3), "dCITL_IC95": ci(dc)})
    return perf, diffs


def curves(y, P, an, etiqueta, modelos=("N2", "N3", "N4")):
    """Observado frente a predicho por cuantiles de riesgo (solo agregados)."""
    rows = []
    for s, msk, q in (("toda la cohorte", np.ones(len(y), bool), 10), ("normalizacion aparente", an, 5)):
        for m in modelos:
            p, yy = P[m][msk], y[msk]
            b = pd.qcut(pd.Series(p).rank(method="first"), q, labels=False)
            for k in range(q):
                w = (b == k).to_numpy()
                rows.append({**etiqueta, "subgrupo": s, "modelo": m, "cuantil": k + 1, "n": int(w.sum()),
                             "eventos": int(yy[w].sum()), "predicho_%": round(100 * p[w].mean(), 3),
                             "observado_%": round(100 * yy[w].mean(), 3)})
    return rows


# ----------------------------------------------------------------------------
# Tabla 1 recalculada (visitas y pacientes unicos; sexo en ambos centros)
# ----------------------------------------------------------------------------
def q(s, d=0):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return ""
    return f"{s.median():.{d}f} ({s.quantile(.25):.{d}f}-{s.quantile(.75):.{d}f})"


def tabla1(c, sitio, coh):
    rows = []
    for g, d in [("Toda la cohorte", c)] + [(k, c[c["grupo"] == k]) for k in ("normalizacion", "estable", "ultimo_anormal")]:
        esi = pd.to_numeric(d["esi"], errors="coerce")
        rows.append({"centro": sitio, "cohorte": coh, "grupo": g, "visitas": len(d), "pacientes_unicos": d["subject_id"].nunique(),
                     "pct_visitas_cohorte": round(100 * len(d) / len(c), 1),
                     "edad_mediana_RIC": q(d["age"]), "mujeres_%": round(100 * (d["sexo"] == "F").sum() / d["sexo"].notna().sum(), 1),
                     "sexo_faltante": int(d["sexo"].isna().sum()),
                     "ESI_1_2_%": round(100 * esi.le(2).sum() / esi.notna().sum(), 1), "ESI_faltante": int(esi.isna().sum()),
                     "llegada_EMS_%": round(100 * (d["llegada"] == "ems").mean(), 1),
                     "sets_mediana_RIC": q(d["n_sets"]), "NEWS2_triage_mediana_RIC": q(d["news2_first"]),
                     "NEWS2_maximo_mediana_RIC": q(d["news2_max"]), "NEWS2_final_mediana_RIC": q(d["news2_last"]),
                     "estadia_h_mediana_RIC": q(d["ed_los_h"], 1), "h_ultimo_set_mediana_RIC": q(d["hours_last_set_to_exit"], 2),
                     "eventos": int(d["y"].sum()), "desenlace_%": round(100 * d["y"].mean(), 2)})
    return rows


# ----------------------------------------------------------------------------
# Ejecucion
# ----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
print("Cargando cohortes y particiones...")
mi = {"A": load_any(S_MI, "cohorte_A"), "B": load_any(S_MI, "cohorte_B")}
mc = {"A": load_any(S_MC, "mcmed_cohorte_A"), "B*": load_any(S_MC, "mcmed_cohorte_B")}
pats = pd.read_csv(os.path.join(FOLDER, "patients.csv.gz"), usecols=["subject_id", "anchor_year_group"])
mi_test_subj = set(pats.loc[pd.to_numeric(pats["anchor_year_group"].astype(str).str[:4], errors="coerce") >= 2017, "subject_id"])
split_mc = {}
for s_ in ("train", "val", "test"):
    for csn in pd.read_csv(os.path.join(MC, f"split_chrono_{s_}.csv"), header=None)[0]:
        split_mc[csn] = s_

R = {k: [] for k in ("muestra", "desempeno", "diferencias", "curvas", "tabla1")}
for sitio, cohs in (("MIMIC-IV-ED", mi), ("MC-MED", mc)):
    for coh, c0 in cohs.items():
        print(f"\n### {sitio} cohorte {coh}")
        c = prep(c0)
        R["tabla1"] += tabla1(c, sitio, coh)
        cols = ["y", "subject_id", "age", "news2_last", "news2_prior_max"] + BASE_CAT + PROC
        d = c.dropna(subset=cols).reset_index(drop=True)
        y, g, an = d["y"].to_numpy(), d["subject_id"].to_numpy(), d["an"].to_numpy()
        R["muestra"].append({"centro": sitio, "cohorte": coh, "n_cohorte": len(c), "n_analizado": len(d),
                             "excluidos_por_faltantes": len(c) - len(d), "eventos": int(y.sum()),
                             "n_normalizacion": int(an.sum()), "eventos_normalizacion": int(y[an].sum())})

        print("  validacion cruzada agrupada + bootstrap...")
        P = oof_predictions(d)
        et = {"centro": sitio, "cohorte": coh, "evaluacion": "VC 5 particiones por paciente"}
        p_, d_ = evaluate(y, P, g, an, rng, et)
        R["desempeno"] += p_; R["diferencias"] += d_; R["curvas"] += curves(y, P, an, et)

        if sitio == "MIMIC-IV-ED":
            te_mask = d["subject_id"].isin(mi_test_subj); tr_mask = ~te_mask
        else:
            sp = d["stay_id"].map(split_mc)
            tr_mask = sp.isin(["train", "val"])
            te_mask = sp.eq("test") & ~d["subject_id"].isin(set(d.loc[tr_mask, "subject_id"]))
        tr, te = d[tr_mask], d[te_mask].reset_index(drop=True)
        print(f"  temporal: entrenamiento {len(tr):,} / prueba {len(te):,} ({int(te['y'].sum())} eventos; "
              f"{int(te['an'].sum())} con normalizacion, {int(te.loc[te['an'], 'y'].sum())} eventos)")
        Ph = holdout_predictions(tr, te)
        et3 = {"centro": sitio, "cohorte": coh, "evaluacion": "particion temporal (sin pacientes compartidos)"}
        yt, at = te["y"].to_numpy(), te["an"].to_numpy()
        p_, d_ = evaluate(yt, Ph, te["subject_id"].to_numpy(), at, rng, et3)
        R["desempeno"] += p_; R["diferencias"] += d_; R["curvas"] += curves(yt, Ph, at, et3)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 40)
titulos = {"muestra": "Muestra comun para N2/N3/N2P/N4",
           "desempeno": "Desempeno y calibracion con IC95% (bootstrap agrupado por paciente, 1.000 replicas)",
           "diferencias": "Diferencias pareadas (N3-N2; sensibilidad N4-N2P; N4-N3)",
           "tabla1": "Tabla 1 recalculada (visitas y pacientes unicos)"}
for k, df in R.items():
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(OUT7, f"fase7c_{k}.csv"), index=False)
    if k in titulos:
        print(f"\n=== {titulos[k]} ===")
        if k == "desempeno":
            df = df.drop(columns=["AUPRC", "AUPRC_IC95"])
        print(df.to_string(index=False))
print("\n(Curvas de calibracion guardadas en fase7c_curvas.csv para la Figura 4)")
print(f"Guardado en {OUT7}")
