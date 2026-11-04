"""
FALSE-STABILITY - Fase 2: construccion de cohortes y medicion del fenomeno.

Corre localmente sobre MIMIC-IV-ED (2.2) + MIMIC-IV (3.1).
NO subas las filas de salida a servicios de IA en linea (acuerdo de uso de PhysioNet).
Solo el resumen agregado que imprime al final es compartible.

Uso:
    python false_stability_cohort.py \
        --ed   ruta/a/mimic-iv-ed/ed \
        --hosp ruta/a/mimiciv/hosp \
        --icu  ruta/a/mimiciv/icu \
        --out  salida

Requisitos: pandas, numpy, pyarrow (para parquet).
"""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed", required=True)
    ap.add_argument("--hosp", required=True)
    ap.add_argument("--icu", required=True)
    ap.add_argument("--out", default="salida")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Cargando tablas...")
    ed, tri, vs, pat, adm, trf, icu = load(args)
    ed = add_age(ed, pat)

    print("Construyendo trayectorias de signos vitales...")
    long = build_vitals_long(ed, tri, vs)
    feats = trajectory_features(long)
    feats["hours_last_set_to_exit"] = (
        ed.set_index("stay_id")["outtime"].reindex(feats["stay_id"]).values
        - feats["last_set_time"].values) / np.timedelta64(1, "h")

    # Variables estaticas de triage
    tri_static = tri[["stay_id", "acuity", "pain", "chiefcomplaint"]]
    feats = feats.merge(tri_static, on="stay_id", how="left")

    flow = []
    print("Cohorte A...")
    a = cohort_a(ed, feats, flow)
    print("Cohorte B...")
    b = cohort_b(ed, feats, adm, trf, icu, flow)

    for c in (a, b):
        c["ed_los_h"] = (c["outtime"] - c["intime"]).dt.total_seconds() / 3600
        c["sets_per_hour"] = c["n_sets"] / c["ed_los_h"].clip(lower=0.25)

    for name, df in (("cohorte_A", a), ("cohorte_B", b)):
        try:
            df.to_parquet(os.path.join(args.out, name + ".parquet"), index=False)
        except ImportError:
            df.to_csv(os.path.join(args.out, name + ".csv.gz"), index=False)

    flow_df = pd.DataFrame(flow, columns=["cohorte", "paso", "n"])
    summ = pd.concat([summarize(a, "A: reconsulta 72h"), summarize(b, "B: UCI 48h")])
    flow_df.to_csv(os.path.join(args.out, "flujo_seleccion.csv"), index=False)
    summ.to_csv(os.path.join(args.out, "resumen_fenomeno.csv"), index=False)

    pd.set_option("display.width", 140)
    print("\n=== Flujo de seleccion ===")
    print(flow_df.to_string(index=False))
    print("\n=== Fenomeno de falsa estabilidad ===")
    print(summ.to_string(index=False))
    print(f"\nArchivos guardados en: {args.out}/")


if __name__ == "__main__":
    main()
