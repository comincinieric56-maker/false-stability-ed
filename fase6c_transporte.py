# ============================================================================
# FALSE-STABILITY - Fase 6c: transporte de modelos MIMIC -> MC-MED (GPU)
#   Celda AUTONOMA: pensada para un entorno recien cambiado a GPU (todo borrado).
#   Monta Drive, lee MIMIC y MC-MED, entrena en MIMIC y aplica SIN CAMBIOS a
#   MC-MED. Necesita haber corrido antes (en cualquier sesion) la fase 6b,
#   que dejo las cohortes y los sets de MC-MED en la carpeta salida_mcmed.
#   Variables armonizadas (presentes en ambas bases): edad, sexo, ESI, llegada,
#   visitas previas a urgencia, ultimo set, trayectoria y proceso de atencion.
#   Se excluyen motivo de consulta, dolor, hospitalizaciones previas y
#   tratamiento (se registran distinto en cada hospital).
#   ~25-35 minutos con GPU. Salida solo agregada.
# ============================================================================
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as _e:
    print("(Drive no montado automaticamente:", _e, ")")

# ---- funciones de cohortes ----


import argparse
import os

import numpy as np
import pandas as pd

VITALS = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature"]

# Rangos fisiologicamente plausibles; fuera de ellos -> NaN
PLAUSIBLE = {
    "heartrate": (20, 250),
    "resprate": (4, 60),
    "o2sat": (50, 100),
    "sbp": (50, 300),
    "dbp": (20, 200),
    "temperature": (30, 43),  # en Celsius, tras convertir
}

ABNORMAL_NEWS = 3     # set anormal: NEWS2 modificado >= 3 (sensibilidad: 5)
NORMAL_NEWS = 1       # ultimo set normal: NEWS2 modificado <= 1
MIN_PARAMS = 4        # un set es valido si tiene >= 4 de los 5 parametros NEWS2
REVISIT_H = 72
ICU_H = 48


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def read(path_dir, name, **kw):
    for ext in (".csv.gz", ".csv"):
        p = os.path.join(path_dir, name + ext)
        if os.path.exists(p):
            return pd.read_csv(p, **kw)
    raise FileNotFoundError(f"No encontre {name}.csv(.gz) en {path_dir}")


def load(args):
    ed = read(args.ed, "edstays", parse_dates=["intime", "outtime"])
    tri = read(args.ed, "triage")
    vs = read(args.ed, "vitalsign", parse_dates=["charttime"])
    pat = read(args.hosp, "patients")
    adm = read(args.hosp, "admissions", parse_dates=["admittime", "dischtime", "deathtime"])
    trf = read(args.hosp, "transfers", parse_dates=["intime", "outtime"])
    icu = read(args.icu, "icustays", parse_dates=["intime", "outtime"])
    return ed, tri, vs, pat, adm, trf, icu


# ----------------------------------------------------------------------------
# Signos vitales y NEWS2 modificado (sin O2 suplementario ni conciencia)
# ----------------------------------------------------------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def clean_vitals(df):
    df = df.copy()
    for v in VITALS:
        df[v] = to_num(df[v])
    # MIMIC-IV-ED registra temperatura en Fahrenheit
    t = df["temperature"]
    df["temperature"] = np.where(t > 50, (t - 32) * 5 / 9, t)
    for v, (lo, hi) in PLAUSIBLE.items():
        df.loc[(df[v] < lo) | (df[v] > hi), v] = np.nan
    return df


def score(x, bins, pts):
    """bins: limites superiores inclusivos; pts: puntaje por tramo (len = len(bins)+1)."""
    out = pd.Series(np.nan, index=x.index)
    ok = x.notna()
    idx = np.searchsorted(np.array(bins), x[ok].values, side="left")
    out[ok] = np.array(pts)[idx]
    return out


def news2_modified(df):
    df = df.copy()
    df["s_rr"] = score(df["resprate"], [8, 11, 20, 24], [3, 1, 0, 2, 3])
    df["s_spo2"] = score(df["o2sat"], [91, 93, 95], [3, 2, 1, 0])
    df["s_sbp"] = score(df["sbp"], [90, 100, 110, 219], [3, 2, 1, 0, 3])
    df["s_hr"] = score(df["heartrate"], [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    df["s_temp"] = score(df["temperature"].round(1), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    comps = ["s_rr", "s_spo2", "s_sbp", "s_hr", "s_temp"]
    df["n_params"] = df[comps].notna().sum(axis=1)
    df["news2"] = df[comps].sum(axis=1, min_count=1)
    df["max_comp"] = df[comps].max(axis=1)
    df["valid_set"] = df["n_params"] >= MIN_PARAMS
    return df


def build_vitals_long(ed, tri, vs):
    tri = tri.merge(ed[["stay_id", "intime"]], on="stay_id", how="inner")
    tri = tri.rename(columns={"intime": "charttime"})
    tri["source"] = "triage"
    vs = vs.copy()
    vs["source"] = "vitalsign"
    cols = ["stay_id", "charttime", "source"] + VITALS
    long = pd.concat([tri[cols], vs[cols]], ignore_index=True)
    long = long.merge(ed[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    # Solo informacion previa a la salida de urgencia (evita fuga)
    long = long[(long["charttime"] >= long["intime"] - pd.Timedelta(minutes=30))
                & (long["charttime"] <= long["outtime"])]
    long = clean_vitals(long)
    long = news2_modified(long)
    long = long[long["valid_set"]].sort_values(["stay_id", "charttime"])
    return long


def trajectory_features(long):
    g = long.groupby("stay_id", sort=False)
    last = g.tail(1).set_index("stay_id")
    first = g.head(1).set_index("stay_id")

    # Sets previos al ultimo
    long = long.copy()
    long["is_last"] = ~long.duplicated("stay_id", keep="last")
    prior = long[~long["is_last"]]
    prior_abn = (prior["news2"] >= ABNORMAL_NEWS) | (prior["max_comp"] >= 3)
    prior_flags = prior.assign(abn=prior_abn).groupby("stay_id")["abn"].any()
    prior_max = prior.groupby("stay_id")["news2"].max()

    gaps = long.groupby("stay_id")["charttime"].diff().dt.total_seconds() / 3600

    f = pd.DataFrame(index=last.index)
    f["n_sets"] = g.size()
    f["news2_first"] = first["news2"]
    f["news2_last"] = last["news2"]
    f["news2_max"] = g["news2"].max()
    f["news2_prior_max"] = prior_max
    f["news2_drop"] = f["news2_max"] - f["news2_last"]
    f["maxcomp_last"] = last["max_comp"]
    f["prior_abnormal"] = prior_flags.reindex(f.index).fillna(False).astype(bool)
    f["last_normal"] = (last["news2"] <= NORMAL_NEWS) & (last["max_comp"] < 2)
    f["last_set_time"] = last["charttime"]
    f["max_gap_h"] = gaps.groupby(long["stay_id"]).max()
    for v in ["heartrate", "resprate", "o2sat", "sbp", "temperature"]:
        f[f"{v}_last"] = last[v]
        f[f"{v}_max"] = g[v].max()
        f[f"{v}_min"] = g[v].min()
    f["shock_index_last"] = last["heartrate"] / last["sbp"]
    f["apparent_normalization"] = f["prior_abnormal"] & f["last_normal"]
    f["stable_throughout"] = (~f["prior_abnormal"]) & f["last_normal"]
    return f.reset_index()


# ----------------------------------------------------------------------------
# Demografia
# ----------------------------------------------------------------------------
def add_age(ed, pat):
    ed = ed.merge(pat[["subject_id", "anchor_age", "anchor_year", "dod"]], on="subject_id", how="left")
    ed["age"] = ed["anchor_age"] + (ed["intime"].dt.year - ed["anchor_year"])
    ed["dod"] = pd.to_datetime(ed["dod"], errors="coerce")
    return ed


# ----------------------------------------------------------------------------
# Cohorte A: alta desde urgencia -> reconsulta con hospitalizacion <= 72 h
# ----------------------------------------------------------------------------
def outcome_revisit(ed_all, idx):
    """Para cada visita indice, busca la siguiente visita a urgencia del mismo paciente."""
    nxt = ed_all[["subject_id", "stay_id", "intime", "hadm_id"]].rename(
        columns={"stay_id": "next_stay_id", "intime": "next_intime", "hadm_id": "next_hadm_id"})
    m = idx[["stay_id", "subject_id", "outtime"]].merge(nxt, on="subject_id", how="left")
    m = m[(m["next_intime"] > m["outtime"])
          & (m["next_intime"] <= m["outtime"] + pd.Timedelta(hours=REVISIT_H))]
    m = m.sort_values("next_intime").drop_duplicates("stay_id")
    out = idx[["stay_id"]].merge(m[["stay_id", "next_stay_id", "next_hadm_id", "next_intime"]],
                                  on="stay_id", how="left")
    out["revisit_72h_any"] = out["next_stay_id"].notna()
    out["revisit_72h_admit"] = out["next_hadm_id"].notna()
    return out[["stay_id", "revisit_72h_any", "revisit_72h_admit"]]


def cohort_a(ed, feats, flow):
    c = ed.copy()
    flow.append(("A", "Visitas a urgencia", len(c)))
    c = c[c["disposition"].astype(str).str.upper() == "HOME"]
    flow.append(("A", "Alta a domicilio", len(c)))
    c = c[c["age"] >= 18]
    flow.append(("A", "Edad >= 18", len(c)))
    c = c.merge(feats, on="stay_id", how="inner")
    c = c[c["n_sets"] >= 2]
    flow.append(("A", ">= 2 sets validos de signos vitales", len(c)))

    y = outcome_revisit(ed, c)
    c = c.merge(y, on="stay_id", how="left")
    c["death_72h"] = (c["dod"].notna()
                      & (c["dod"] <= c["outtime"].dt.normalize() + pd.Timedelta(days=3)))
    c["outcome"] = c["revisit_72h_admit"]
    return c


# ----------------------------------------------------------------------------
# Cohorte B: hospitalizacion en sala -> UCI <= 48 h
# ----------------------------------------------------------------------------
INTERMEDIATE = r"intermediate|step ?down|stepdown"


def first_unit_after_ed(trf):
    t = trf.dropna(subset=["hadm_id"]).copy()
    t = t[t["eventtype"].astype(str).str.lower() != "discharge"]
    t["cu"] = t["careunit"].astype(str)
    t = t[~t["cu"].str.contains("Emergency", case=False, na=False)]
    t = t.sort_values(["hadm_id", "intime"]).drop_duplicates("hadm_id")
    return t[["hadm_id", "cu", "intime"]].rename(columns={"cu": "first_unit", "intime": "ward_in"})


def cohort_b(ed, feats, adm, trf, icu, flow):
    c = ed[ed["hadm_id"].notna()].copy()
    flow.append(("B", "Visitas con hospitalizacion", len(c)))
    c = c[c["age"] >= 18]
    flow.append(("B", "Edad >= 18", len(c)))

    fu = first_unit_after_ed(trf)
    c = c.merge(fu, on="hadm_id", how="inner")

    icu_first = icu.sort_values("intime").drop_duplicates("hadm_id")[["hadm_id", "intime", "first_careunit"]]
    icu_first = icu_first.rename(columns={"intime": "icu_in"})
    c = c.merge(icu_first, on="hadm_id", how="left")

    direct_icu = c["icu_in"].notna() & (c["icu_in"] <= c["ward_in"] + pd.Timedelta(hours=1))
    intermediate = c["first_unit"].str.contains(INTERMEDIATE, case=False, na=False)
    periop = c["first_unit"].str.contains("PACU|Operating", case=False, na=False)
    c = c[~direct_icu & ~intermediate & ~periop]
    flow.append(("B", "Primera unidad: sala convencional", len(c)))

    c = c.merge(feats, on="stay_id", how="inner")
    c = c[c["n_sets"] >= 2]
    flow.append(("B", ">= 2 sets validos de signos vitales", len(c)))

    win = pd.Timedelta(hours=ICU_H)
    c["icu_48h"] = c["icu_in"].notna() & (c["icu_in"] <= c["ward_in"] + win)
    c = c.merge(adm[["hadm_id", "deathtime"]], on="hadm_id", how="left")
    c["death_ward_48h"] = (~c["icu_48h"] & c["deathtime"].notna()
                           & (c["deathtime"] <= c["ward_in"] + win))
    c["outcome"] = c["icu_48h"] | c["death_ward_48h"]
    return c


# ----------------------------------------------------------------------------
# Resumen agregado (lo unico que conviene compartir)
# ----------------------------------------------------------------------------
def summarize(c, name):
    rows = []
    groups = {
        "Toda la cohorte": pd.Series(True, index=c.index),
        "Normalizacion aparente": c["apparent_normalization"],
        "Estable desde triage": c["stable_throughout"],
        "Ultimo set anormal": ~c["last_normal"],
    }
    for label, mask in groups.items():
        sub = c[mask]
        n, ev = len(sub), int(sub["outcome"].sum())
        rows.append({
            "cohorte": name, "grupo": label, "n": n,
            "pct_cohorte": round(100 * n / max(len(c), 1), 1),
            "eventos": ev, "tasa_pct": round(100 * ev / max(n, 1), 2),
        })
    return pd.DataFrame(rows)



# ---- fase 3 ----
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



# ---- fase 5 ----
# ============================================================================
# FALSE-STABILITY - Fase 5: red de secuencia M3 (GPU)
#   Rama de secuencia: GRU sobre cada set de signos vitales (valores, puntajes
#   NEWS2, mascaras de faltantes y tiempos irregulares).
#   Rama tabular: FT-Transformer con estaticas, historia y tratamiento.
#   Fusion -> riesgo. Comparacion contra M2 (gradient boosting de la fase 3).
#
# Requiere: celda de la fase 3 corrida en esta sesion, y el archivo
# /content/false_stability_cohort.py (lo crea la celda %%writefile de la fase 2).
# Entorno de ejecucion -> Cambiar tipo de entorno -> GPU (T4 basta).
# Todo lo que imprime y los graficos son agregados y se pueden compartir.
# ============================================================================
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
L_MAX = 24            # ultimos 24 sets por visita (el p95 esta muy por debajo)
D = 64                # dimension oculta
N_SEEDS = 3           # semillas del modelo completo (se promedian)
EPOCHS = 40
PATIENCE = 6
BATCH = 1024
LR = 1e-3
COHORTES = [("cohorte_B", "B: UCI 48h", True),          # (archivo, nombre, correr ablaciones)
            ("cohorte_A", "A: reconsulta 72h", False)]
SEQ_ARCH = "gru"      # "gru" o "transformer"

SEQ_VITALS = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature"]
SEQ_SCORES = ["s_rr", "s_spo2", "s_sbp", "s_hr", "s_temp", "news2"]
SEQ_TIME = ["t_arr", "t_exit", "dt_prev"]
SEQ_CONT = SEQ_VITALS + SEQ_SCORES + SEQ_TIME
SEQ_FLAGS = [v + "_m" for v in SEQ_VITALS] + ["is_triage"]
SEQ_CH = SEQ_CONT + SEQ_FLAGS
TIME_IDX = [SEQ_CH.index(c) for c in SEQ_TIME]
TAB_NUM = STATIC_NUM + TX_NUM


# ----------------------------------------------------------------------------
# Secuencias
# ----------------------------------------------------------------------------
def build_long_seq():
    print("Reconstruyendo secuencias de signos vitales...")
    tri = read_from_zip("triage")
    vs = read_from_zip("vitalsign", parse_dates=["charttime"])
    long = build_vitals_long(ed_all, tri, vs)
    h = lambda td: td.dt.total_seconds() / 3600
    long["t_arr"] = np.log1p(h(long["charttime"] - long["intime"]).clip(lower=0))
    long["t_exit"] = np.log1p(h(long["outtime"] - long["charttime"]).clip(lower=0))
    long["dt_prev"] = np.log1p(h(long.groupby("stay_id")["charttime"].diff()).fillna(0).clip(lower=0))
    long["is_triage"] = (long["source"] == "triage").astype(float)
    for v in SEQ_VITALS:
        long[v + "_m"] = long[v].isna().astype(float)
    long = long.sort_values(["stay_id", "charttime"]).groupby("stay_id").tail(L_MAX)
    return long[["stay_id"] + SEQ_CH]


def seq_stats(long, stay_ids):
    sub = long[long["stay_id"].isin(stay_ids)]
    return sub[SEQ_CONT].mean(), sub[SEQ_CONT].std().replace(0, 1)


def build_seq(c, long, stats):
    """Devuelve X [N, L_MAX, F] (alineado a la izquierda) y largos [N], en el orden de c."""
    mu, sd = stats
    sub = long[long["stay_id"].isin(c["stay_id"])].copy()
    sub[SEQ_CONT] = ((sub[SEQ_CONT] - mu) / sd).fillna(0)
    row = pd.Categorical(sub["stay_id"], categories=c["stay_id"].to_numpy()).codes
    pos = sub.groupby("stay_id").cumcount().to_numpy()
    X = np.zeros((len(c), L_MAX, len(SEQ_CH)), dtype=np.float32)
    X[row, pos, :] = sub[SEQ_CH].to_numpy(np.float32)
    lengths = np.bincount(row, minlength=len(c)).astype(np.int64)
    return X, np.maximum(lengths, 1)


def tab_encoders(tr):
    med = tr[TAB_NUM].median()
    mu = tr[TAB_NUM].fillna(med).mean()
    sd = tr[TAB_NUM].fillna(med).std().replace(0, 1)
    cats = {c: {v: i + 1 for i, v in enumerate(tr[c].astype(str).unique())} for c in CAT}  # 0 = desconocido
    return med, mu, sd, cats


def build_tab(c, enc):
    med, mu, sd, cats = enc
    xn = ((c[TAB_NUM].fillna(med) - mu) / sd).to_numpy(np.float32)
    xc = np.stack([c[k].astype(str).map(cats[k]).fillna(0).to_numpy(np.int64) for k in CAT], 1)
    return xn, xc


def split_val(tr, frac=0.15, seed=SEED):
    subj = tr["subject_id"].unique()
    val_subj = set(np.random.default_rng(seed).choice(subj, int(frac * len(subj)), replace=False))
    return tr["subject_id"].isin(val_subj).to_numpy()


# ============================================================================
# MODELO
# ============================================================================
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "(sin GPU: sera lento)")


class FTTabular(nn.Module):
    """FT-Transformer liviano: un token por variable + CLS."""
    def __init__(self, n_num, cat_cards, d=D, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.embs = nn.ModuleList([nn.Embedding(k, d) for k in cat_cards])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, xn, xc):
        tn = xn.unsqueeze(-1) * self.w + self.b
        tc = torch.stack([e(xc[:, i]) for i, e in enumerate(self.embs)], 1)
        t = torch.cat([self.cls.expand(xn.shape[0], -1, -1), tn, tc], 1)
        return self.norm(self.enc(t)[:, 0])


class SeqEncoder(nn.Module):
    def __init__(self, f, d=D, arch=SEQ_ARCH, drop=0.1):
        super().__init__()
        self.arch = arch
        self.inp = nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Dropout(drop))
        if arch == "gru":
            self.rnn = nn.GRU(d, d, num_layers=1, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, lengths):
        h = self.inp(x)
        if self.arch == "gru":
            packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hn = self.rnn(packed)
            return self.norm(hn[-1])
        mask = torch.arange(x.shape[1], device=x.device)[None, :] >= lengths[:, None]
        z = self.enc(h, src_key_padding_mask=mask)
        last = z[torch.arange(len(z), device=z.device), lengths - 1]
        mean = (z * (~mask).unsqueeze(-1)).sum(1) / lengths[:, None]
        return self.norm(last + mean)


class M3(nn.Module):
    def __init__(self, f_seq, n_num, cat_cards, use_seq=True, use_tab=True, d=D, drop=0.2):
        super().__init__()
        self.use_seq, self.use_tab = use_seq, use_tab
        self.seq = SeqEncoder(f_seq, d) if use_seq else None
        self.tab = FTTabular(n_num, cat_cards, d) if use_tab else None
        k = d * (int(use_seq) + int(use_tab))
        self.head = nn.Sequential(nn.Linear(k, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, xs, ln, xn, xc):
        parts = []
        if self.use_seq:
            parts.append(self.seq(xs, ln))
        if self.use_tab:
            parts.append(self.tab(xn, xc))
        return self.head(torch.cat(parts, 1)).squeeze(-1)


def to_dev(*arrs):
    return [torch.as_tensor(a).to(DEVICE) for a in arrs]


@torch.no_grad()
def predict(model, T, bs=4096):
    model.eval()
    xs, ln, xn, xc = T
    out = []
    for i in range(0, len(ln), bs):
        out.append(torch.sigmoid(model(xs[i:i + bs], ln[i:i + bs], xn[i:i + bs], xc[i:i + bs])).float().cpu())
    return torch.cat(out).numpy()


def train_one(Tfit, yfit, Tval, yval, cards, seed, use_seq=True, use_tab=True, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    model = M3(len(SEQ_CH), len(TAB_NUM), cards, use_seq, use_tab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.as_tensor(yfit, dtype=torch.float32, device=DEVICE)
    n = len(yfit)
    best, best_state, bad = -1, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            logit = model(Tfit[0][idx], Tfit[1][idx], Tfit[2][idx], Tfit[3][idx])
            loss = lossf(logit, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ap = average_precision_score(yval, predict(model, Tval))
        sched.step(ap)
        if ap > best + 1e-4:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    print(f"    {tag} semilla {seed}: {ep + 1} epocas, AUPRC val {best:.4f}, {time.time() - t0:.0f}s")
    return model


def eval_rows(y, preds, nombre, subset, ref="M2_gbm"):
    ci = boot_ci(y, preds, ref=ref)
    rows = []
    for k, p in preds.items():
        m = metrics(y, p)
        rows.append({"cohorte": nombre, "subgrupo": subset, "modelo": k, "n": len(y), "eventos": int(y.sum()),
                     "AUROC": round(m["AUROC"], 3), "AUROC_IC95": ci[k]["AUROC_ic"],
                     "AUPRC": round(m["AUPRC"], 3), "AUPRC_IC95": ci[k]["AUPRC_ic"],
                     "O/E": round(m["O/E"], 2), "pendiente_cal": round(m["pendiente"], 2),
                     f"dAUPRC_vs_{ref}": ci[k].get(f"dAUPRC_vs_{ref}", "")})
    return rows


def temporal_attribution(model, T, mask_rows, y, nombre):
    """Gradiente x entrada sobre la secuencia: que posicion (desde el final) pesa mas."""
    model.eval()
    xs, ln, xn, xc = T
    idx = torch.as_tensor(np.where(mask_rows)[0], device=DEVICE)
    pos_share, ch_share = [], []
    for i in range(0, len(idx), 2048):
        b = idx[i:i + 2048]
        x = xs[b].clone().requires_grad_(True)
        model.zero_grad()
        # cuDNN no permite backward de una GRU en modo eval: se desactiva solo aqui
        with torch.backends.cudnn.flags(enabled=False):
            model(x, ln[b], xn[b], xc[b]).sum().backward()
        a = (x.grad * x).abs().detach()                      # [B, L, F]
        per_pos = a.sum(-1)
        per_pos = per_pos / per_pos.sum(1, keepdim=True).clamp_min(1e-12)
        pos_share.append(per_pos.cpu().numpy())
        per_ch = a.sum(1)
        ch_share.append((per_ch / per_ch.sum(1, keepdim=True).clamp_min(1e-12)).cpu().numpy())
    P = np.concatenate(pos_share)
    C = np.concatenate(ch_share)
    L = ln[idx].cpu().numpy()
    yy = y[mask_rows]

    labels = []
    for l in L:
        lab = np.array(["intermedios"] * L_MAX, dtype=object)
        lab[l - 1] = "ultimo set"
        if l >= 2: lab[l - 2] = "penultimo"
        if l >= 3: lab[l - 3] = "antepenultimo"
        lab[0] = "triage / primer set" if l > 1 else "ultimo set"
        lab[l:] = "padding"
        labels.append(lab)
    labels = np.stack(labels)
    order = ["ultimo set", "penultimo", "antepenultimo", "intermedios", "triage / primer set"]
    rows = []
    for grp, sel in (("todos", np.ones(len(yy), bool)), ("con desenlace", yy == 1), ("sin desenlace", yy == 0)):
        r = {"cohorte": nombre, "grupo": grp, "n": int(sel.sum())}
        for o in order:
            r[o] = round(100 * (P[sel] * (labels[sel] == o)).sum(1).mean(), 1)
        rows.append(r)
    pos = pd.DataFrame(rows)

    groups = {"valores vitales": SEQ_VITALS, "puntajes NEWS2": SEQ_SCORES, "tiempos": SEQ_TIME,
              "faltantes/triage": SEQ_FLAGS}
    crow = []
    for grp, sel in (("con desenlace", yy == 1), ("sin desenlace", yy == 0)):
        r = {"cohorte": nombre, "grupo": grp}
        for g, chs in groups.items():
            r[g] = round(100 * C[sel][:, [SEQ_CH.index(c) for c in chs]].sum(1).mean(), 1)
        crow.append(r)
    return pos, pd.DataFrame(crow)



# ============================================================================
# FASE 6c: TRANSPORTE
# ============================================================================
import re as _re
from sklearn.linear_model import LogisticRegression

MC = os.path.join(FOLDER, "mcmed")
OUT_MC = os.path.join(FOLDER, "salida_mcmed")
N_SEEDS_6C = 3
for _f in ("mcmed_cohorte_A.csv.gz", "mcmed_cohorte_B.csv.gz", "mcmed_sets.csv.gz"):
    if not os.path.exists(os.path.join(OUT_MC, _f)):
        raise RuntimeError(f"Falta {_f} en salida_mcmed: corre primero la fase 6b.")

_YEAR_RE = _re.compile(r"\b(2[1-4]\d\d)\b")


def fix_time(s):
    s = s.astype("string")
    fixed = s.str.replace(_YEAR_RE, lambda m: str(int(m.group(1)) - 400), regex=True)
    t = pd.to_datetime(fixed, format="ISO8601", errors="coerce")
    if t.notna().mean() < 0.99 * s.notna().mean():
        t = t.fillna(pd.to_datetime(fixed, format="mixed", errors="coerce"))
    return t


# --- Variables armonizadas ---
CAT_H = ["sexo", "esi", "llegada"]
TAB_H = ["age", "prev_ed_180d", "prev_ed_365d"]
PROC = ["n_sets", "max_gap_h", "hours_last_set_to_exit", "ed_los_h", "sets_per_hour"]
NUM_H = TAB_H + LAST_NUM + TRAJ_NUM          # TRAJ_NUM ya incluye las de proceso
SPEC_M2H = dict(num=NUM_H, cat=CAT_H)
SPEC_M2H_NOPROC = dict(num=[f for f in NUM_H if f not in PROC], cat=CAT_H)


def llegada_h(s):
    s = s.fillna("").astype(str).str.upper()
    return np.select([s.str.contains("WALK|SELF"), s.str.contains("AMBUL|EMS|HELI")], ["propio", "ems"], "otro")


def harmonize(c):
    c = c.copy()
    c["sexo"] = c["gender"].where(c["gender"].isin(["F", "M"]), "NA").astype(str)
    c["esi"] = pd.to_numeric(c["acuity"], errors="coerce").fillna(-1).astype(int).astype(str).replace("-1", "NA")
    c["llegada"] = llegada_h(c["arrival_transport"])
    return c


# --- MIMIC ---
print("\n### MIMIC: cohortes y secuencias")
long_seq = build_long_seq()
mimic = {k: harmonize(prep(load_out(f))).reset_index(drop=True)
         for k, f in (("A", "cohorte_A"), ("B", "cohorte_B"))}

# --- MC-MED ---
print("\n### MC-MED: visitas, sets y variables")
v = pd.read_csv(os.path.join(MC, "visits.csv"), low_memory=False)
v["intime"], v["outtime"] = fix_time(v["Arrival_time"]), fix_time(v["Departure_time"])
v = v.rename(columns={"CSN": "stay_id", "MRN": "subject_id"})

ev = v[["subject_id", "stay_id", "intime"]].dropna().sort_values(["subject_id", "intime"]).reset_index(drop=True)
ev["one"] = 1
hist_mc = ev[["stay_id"]].copy()
for d in (180, 365):
    r = ev.set_index("intime").groupby("subject_id")["one"].rolling(f"{d}D", closed="left").sum().reset_index(drop=True)
    hist_mc[f"prev_ed_{d}d"] = r.fillna(0).values

sets = pd.read_csv(os.path.join(OUT_MC, "mcmed_sets.csv.gz"), parse_dates=["charttime"])
sets = sets.merge(v[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
sets = news2_modified(clean_vitals(sets)).sort_values(["stay_id", "charttime"])
feats_mc = trajectory_features(sets)
feats_mc = feats_mc.merge(v[["stay_id", "outtime"]], on="stay_id")
feats_mc["hours_last_set_to_exit"] = (feats_mc["outtime"] - feats_mc["last_set_time"]).dt.total_seconds() / 3600
feats_mc = feats_mc.drop(columns="outtime")

h = lambda td: td.dt.total_seconds() / 3600
sq = sets.copy()
sq["t_arr"] = np.log1p(h(sq["charttime"] - sq["intime"]).clip(lower=0))
sq["t_exit"] = np.log1p(h(sq["outtime"] - sq["charttime"]).clip(lower=0))
sq["dt_prev"] = np.log1p(h(sq.groupby("stay_id")["charttime"].diff()).fillna(0).clip(lower=0))
sq["is_triage"] = (sq["source"] == "triage").astype(float)
for c_ in SEQ_VITALS:
    sq[c_ + "_m"] = sq[c_].isna().astype(float)
long_seq_mc = sq.groupby("stay_id").tail(L_MAX)[["stay_id"] + SEQ_CH]

KEEP = ["stay_id", "subject_id", "outcome", "age", "gender", "acuity", "arrival_transport", "ED_LOS"]
mcmed = {}
for k in ("A", "B"):
    c = pd.read_csv(os.path.join(OUT_MC, f"mcmed_cohorte_{k}.csv.gz"), low_memory=False)
    c = c[[x for x in KEEP if x in c]].merge(feats_mc, on="stay_id", how="inner").merge(hist_mc, on="stay_id", how="left")
    c = c[c["n_sets"] >= 2]
    c["ed_los_h"] = pd.to_numeric(c["ED_LOS"], errors="coerce")
    c["sets_per_hour"] = c["n_sets"] / c["ed_los_h"].clip(lower=0.25)
    c["y"] = c["outcome"].astype(int)
    c["apparent_normalization_i"] = c["apparent_normalization"].astype(int)
    c["prior_abnormal_i"] = c["prior_abnormal"].astype(int)
    mcmed[k] = harmonize(c).reset_index(drop=True)
    print(f"  MC-MED cohorte {k}: n={len(c):,} eventos={c['y'].sum():,} normalizacion aparente={c['apparent_normalization'].sum():,}")

# Tabular armonizado para M3
TAB_NUM = TAB_H
CAT = CAT_H


def zero_channels(T, names):
    if not names:
        return T
    X = T[0].clone()
    X[:, :, [SEQ_CH.index(n) for n in names]] = 0
    return [X] + list(T[1:])


def platt(p_val, y_val):
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg(p_val).reshape(-1, 1), y_val)
    return lambda p: lr.predict_proba(lg(p).reshape(-1, 1))[:, 1]


VARS_M3 = {"M3h": [], "M3h sin tiempos": SEQ_TIME, "M3h solo ritmo": SEQ_VITALS + SEQ_SCORES}
rows, nombres = [], {"A": "A: reconsulta 72h", "B": "B: UCI 48h (MIMIC) / muerte-hospicio (MC-MED)"}

for k in ("B", "A"):
    print(f"\n### Cohorte {k}: entrenamiento en MIMIC")
    c = mimic[k]
    tr, te = c[c["split"] == "train"].reset_index(drop=True), c[c["split"] == "test"].reset_index(drop=True)
    vm = split_val(tr)
    fitd, vald = tr[~vm].reset_index(drop=True), tr[vm].reset_index(drop=True)
    mc = mcmed[k]

    preds = {"MIMIC prueba": {}, "MC-MED": {}}
    for name, f in (("M0 NEWS2", fit_m0(tr)), ("M2h", fit_gbm(tr, SPEC_M2H)), ("M2h sin proceso", fit_gbm(tr, SPEC_M2H_NOPROC))):
        preds["MIMIC prueba"][name], preds["MC-MED"][name] = f(te), f(mc)
        print(f"  {name} listo")

    stats = seq_stats(long_seq, fitd["stay_id"])
    enc = tab_encoders(fitd)
    cards = [len(enc[3][x]) + 1 for x in CAT]

    def T_of(d, ls):
        X, ln = build_seq(d, ls, stats)
        xn, xc = build_tab(d, enc)
        return to_dev(X, ln, xn, xc)

    Tfit, Tval = T_of(fitd, long_seq), T_of(vald, long_seq)
    Tte, Tmc = T_of(te, long_seq), T_of(mc, long_seq_mc)
    yfit, yval = fitd["y"].to_numpy(), vald["y"].to_numpy()
    for vname, zeros in VARS_M3.items():
        Tf, Tv, Tt, Tm = (zero_channels(T, zeros) for T in (Tfit, Tval, Tte, Tmc))
        pv, pt, pm = [], [], []
        for s in range(N_SEEDS_6C):
            m = train_one(Tf, yfit, Tv, yval, cards, 100 + s, tag=vname)
            pv.append(predict(m, Tv)); pt.append(predict(m, Tt)); pm.append(predict(m, Tm))
            del m
        cal_ = platt(np.mean(pv, 0), yval)
        preds["MIMIC prueba"][vname] = cal_(np.mean(pt, 0))
        preds["MC-MED"][vname] = cal_(np.mean(pm, 0))
        del Tf, Tv, Tt, Tm
        torch.cuda.empty_cache()
    del Tfit, Tval, Tte, Tmc
    torch.cuda.empty_cache()

    for base_name, d in (("MIMIC prueba", te), ("MC-MED", mc)):
        y = d["y"].to_numpy()
        an = d["apparent_normalization"].to_numpy().astype(bool)
        for subset, msk in (("toda la cohorte", np.ones(len(y), bool)), ("normalizacion aparente", an)):
            P = {n: p[msk] for n, p in preds[base_name].items()}
            ci = boot_ci(y[msk], P, ref="M2h")
            for n, p in P.items():
                m_ = metrics(y[msk], p)
                rows.append({"cohorte": nombres[k], "base": base_name, "subgrupo": subset, "modelo": n,
                             "n": int(msk.sum()), "eventos": int(y[msk].sum()),
                             "AUROC": round(m_["AUROC"], 3), "AUROC_IC95": ci[n]["AUROC_ic"],
                             "AUPRC": round(m_["AUPRC"], 3), "AUPRC_IC95": ci[n]["AUPRC_ic"],
                             "O/E": round(m_["O/E"], 2), "dAUPRC_vs_M2h": ci[n].get("dAUPRC_vs_M2h", "")})

res = pd.DataFrame(rows)
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 30)
print("\n=== Transporte MIMIC -> MC-MED (modelos entrenados solo en MIMIC; IC95 bootstrap) ===")
for (coh, sub), g in res.groupby(["cohorte", "subgrupo"], sort=False):
    print(f"\n--- {coh} | {sub} ---")
    print(g.drop(columns=["cohorte", "subgrupo"]).to_string(index=False))
res.to_csv(os.path.join(OUT_MC, "fase6c_transporte.csv"), index=False)
print(f"\nGuardado en {OUT_MC}/fase6c_transporte.csv")
print("Nota: en la cohorte B el desenlace de MC-MED es otro (muerte/hospicio); su O/E no es interpretable.")
