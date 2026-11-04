# ============================================================================
# FALSE-STABILITY - Fase 7a: contrastes epidemiologicos y sensibilidades
# (respuesta a la revision; definiciones fijadas ANTES de ejecutar)
#
#   1. Asociacion principal: MODELO 2 (edad, sexo, ESI, motivo, llegada).
#      Modelo 1 y modelo 3 (+ proceso de atencion) como sensibilidad.
#   2. Contraste directo ajustado: normalizacion aparente vs ultimo set anormal.
#   3. Sesgo por frecuencia de medicion: estratos por numero de sets (+ test de
#      interaccion) y definicion basada SOLO en el triage.
#   4. Definiciones alternativas (previo NEWS2 >=5; ultimo = 0), ajustadas.
#   5. MC-MED: reconstruccion de sets sin relleno y con agrupacion estricta;
#      exclusion de oxigeno suplementario.
#   6. Fenotipos preespecificados (umbrales NEWS2), ambos centros, dentro del
#      subgrupo con normalizacion aparente. Asociativo, no causal.
#
# Todo con errores estandar agrupados por paciente; se reportan n total,
# n analizado (casos completos), eventos y p. CPU, ~15-20 min (incluye releer
# numerics de MC-MED para la sensibilidad de reconstruccion).
# Salida solo agregada, en la carpeta salida_fase7.
# ============================================================================
import os, re, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
RUN_MC_RECON = True   # sensibilidad de reconstruccion en MC-MED (relee numerics, ~8 min)

# --- Definiciones congeladas ---
ABN, NORM = 3, 1
NSETS_BINS = [1, 2, 4, 6, np.inf]
NSETS_LAB = ["2", "3-4", "5-6", ">=7"]
FENOTIPOS = {   # (variable, condicion, etiqueta) ; umbrales de puntaje NEWS2 >= 1-2
    "hipotension": ("sbp_min", lambda s: s <= 100, "PAS minima <=100 mmHg"),
    "hipoxemia": ("o2sat_min", lambda s: s <= 93, "SpO2 minima <=93%"),
    "fiebre": ("temperature_max", lambda s: s >= 38.1, "Temperatura maxima >=38,1 C"),
    "hipotermia": ("temperature_min", lambda s: s <= 36.0, "Temperatura minima <=36,0 C"),
    "taquicardia": ("heartrate_max", lambda s: s >= 111, "FC maxima >=111/min"),
    "taquipnea": ("resprate_max", lambda s: s >= 21, "FR maxima >=21/min"),
    "gravedad_previa": ("news2_prior_max", lambda s: s >= 5, "NEWS2 previo maximo >=5"),
}


def load_any(folder, name):
    for ext, rd in ((".parquet", pd.read_parquet), (".csv.gz", lambda p: pd.read_csv(p, low_memory=False))):
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return rd(p)
    raise FileNotFoundError(f"{name} no esta en {folder}")


# ----------------------------------------------------------------------------
# Preparacion comun
# ----------------------------------------------------------------------------
def prep(c, top_cc=30):
    c = c.copy()
    c["grupo"] = np.select([c["apparent_normalization"].astype(bool), c["stable_throughout"].astype(bool)],
                           ["normalizacion", "estable"], "ultimo_anormal")
    c["y"] = c["outcome"].astype(bool).astype(int)
    c["sexo"] = c["gender"].where(c["gender"].isin(["F", "M"]))          # otros -> faltante
    c["esi"] = pd.to_numeric(c["acuity"], errors="coerce")
    c["esi"] = c["esi"].round().astype("Int64").astype(str).replace("<NA>", np.nan)
    arr = c["arrival_transport"].fillna("").astype(str).str.upper()
    c["llegada"] = np.select([arr.str.contains("WALK|SELF"), arr.str.contains("AMBUL|EMS|HELI")], ["propio", "ems"], "otro")
    cc = c["chiefcomplaint"].fillna("NA").astype(str).str.lower().str.strip()
    c["motivo"] = np.where(cc.isin(cc.value_counts().index[:top_cc]), cc, "otro")
    c["edad10"] = pd.to_numeric(c["age"], errors="coerce") / 10
    c["log_sets"] = np.log(c["n_sets"])
    c["los_h"] = pd.to_numeric(c["ed_los_h"], errors="coerce").clip(0, 72)
    c["h_ult"] = pd.to_numeric(c["hours_last_set_to_exit"], errors="coerce").clip(0, 48)
    c["nsets_cat"] = pd.cut(c["n_sets"], NSETS_BINS, labels=NSETS_LAB).astype(str)
    return c


COV = {"M1": "edad10 + C(sexo) + C(esi)",
       "M2": "edad10 + C(sexo) + C(esi) + C(motivo) + C(llegada)",
       "M3": "edad10 + C(sexo) + C(esi) + C(motivo) + C(llegada) + log_sets + los_h + h_ult"}
NEEDED = {"M1": ["edad10", "sexo", "esi"], "M2": ["edad10", "sexo", "esi", "motivo", "llegada"],
          "M3": ["edad10", "sexo", "esi", "motivo", "llegada", "log_sets", "los_h", "h_ult"]}


def fit(df, formula, cols):
    d = df.dropna(subset=cols + ["y", "subject_id"])
    m = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d["subject_id"].values})
    return m, d


def term(m, k):
    b, se, p = m.params[k], m.bse[k], m.pvalues[k]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p


def fmt_or(o, lo, hi):
    return f"{o:.2f} ({lo:.2f}-{hi:.2f})"


def fmt_p(p):
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def contrasts(c, sitio, coh):
    """Asociacion principal y sensibilidades por modelo; contraste directo vs ultimo anormal."""
    rows = []
    for ref, lado in (("estable", "normalizacion vs estable"), ("ultimo_anormal", "normalizacion vs ultimo set anormal")):
        for mod in ("M2", "M1", "M3"):
            f = f"y ~ C(grupo, Treatment('{ref}')) + {COV[mod]}"
            m, d = fit(c, f, NEEDED[mod])
            k = f"C(grupo, Treatment('{ref}'))[T.normalizacion]"
            o, lo, hi, p = term(m, k)
            rows.append({"centro": sitio, "cohorte": coh, "contraste": lado,
                         "modelo": {"M2": "M2 principal", "M1": "M1 sensibilidad", "M3": "M3 sensibilidad (proceso)"}[mod],
                         "n_total": len(c), "n_analizado": len(d), "eventos": int(d["y"].sum()),
                         "OR (IC95%)": fmt_or(o, lo, hi), "p": fmt_p(p)})
    return rows


def freq_bias(c, sitio, coh):
    """Estratos por numero de sets, test de interaccion y definicion solo-triage."""
    rows = []
    s = c[c["grupo"].isin(["normalizacion", "estable"])].copy()
    for cat in NSETS_LAB:
        d0 = s[s["nsets_cat"] == cat]
        n_exp, n_ref = int((d0["grupo"] == "normalizacion").sum()), int((d0["grupo"] == "estable").sum())
        e_exp = int(d0.loc[d0["grupo"] == "normalizacion", "y"].sum())
        r = {"centro": sitio, "cohorte": coh, "analisis": f"estrato n_sets = {cat}", "n_expuestos": n_exp,
             "eventos_expuestos": e_exp, "n_referencia": n_ref,
             "tasa_exp_%": round(100 * e_exp / max(n_exp, 1), 2),
             "tasa_ref_%": round(100 * d0.loc[d0["grupo"] == "estable", "y"].mean(), 2) if n_ref else np.nan}
        if n_exp >= 50 and e_exp >= 10:
            try:
                m, d = fit(d0, f"y ~ C(grupo, Treatment('estable')) + {COV['M2']}", NEEDED["M2"])
                o, lo, hi, p = term(m, "C(grupo, Treatment('estable'))[T.normalizacion]")
                r.update({"OR M2 (IC95%)": fmt_or(o, lo, hi), "p": fmt_p(p)})
            except Exception as e:
                r.update({"OR M2 (IC95%)": f"no estimable ({type(e).__name__})"})
        else:
            r.update({"OR M2 (IC95%)": "muy pocos eventos"})
        rows.append(r)
    # Interaccion grupo x estrato de sets
    try:
        f = f"y ~ C(grupo, Treatment('estable')) * C(nsets_cat, Treatment('2')) + {COV['M2']}"
        m, d = fit(s, f, NEEDED["M2"] + ["nsets_cat"])
        inter = [k for k in m.params.index if ":" in k]
        R = np.zeros((len(inter), len(m.params)))
        for i, k in enumerate(inter):
            R[i, list(m.params.index).index(k)] = 1
        pint = float(np.asarray(m.wald_test(R, scalar=True).pvalue))
        rows.append({"centro": sitio, "cohorte": coh, "analisis": "test de interaccion grupo x n_sets", "p": fmt_p(pint)})
    except Exception as e:
        rows.append({"centro": sitio, "cohorte": coh, "analisis": "test de interaccion grupo x n_sets",
                     "p": f"no estimable ({type(e).__name__})"})
    # Ajuste adicional por n_sets (categorico)
    m, d = fit(s, f"y ~ C(grupo, Treatment('estable')) + C(nsets_cat) + {COV['M2']}", NEEDED["M2"] + ["nsets_cat"])
    o, lo, hi, p = term(m, "C(grupo, Treatment('estable'))[T.normalizacion]")
    rows.append({"centro": sitio, "cohorte": coh, "analisis": "M2 + categoria de n_sets", "n_expuestos": int((d["grupo"] == "normalizacion").sum()),
                 "eventos_expuestos": int(d.loc[d["grupo"] == "normalizacion", "y"].sum()), "OR M2 (IC95%)": fmt_or(o, lo, hi), "p": fmt_p(p)})
    # Definicion basada solo en el triage (no depende de sets intermedios)
    last_norm = (c["news2_last"] <= NORM) & (c["maxcomp_last"] < 2)
    t = c[last_norm].copy()
    t["grupo_t"] = np.where(t["news2_first"] >= ABN, "triage_anormal", "triage_normal")
    m, d = fit(t, f"y ~ C(grupo_t, Treatment('triage_normal')) + {COV['M2']}", NEEDED["M2"])
    o, lo, hi, p = term(m, "C(grupo_t, Treatment('triage_normal'))[T.triage_anormal]")
    rows.append({"centro": sitio, "cohorte": coh, "analisis": "definicion solo triage (triage NEWS2 >=3 y ultimo normal)",
                 "n_expuestos": int((d["grupo_t"] == "triage_anormal").sum()),
                 "eventos_expuestos": int(d.loc[d["grupo_t"] == "triage_anormal", "y"].sum()),
                 "n_referencia": int((d["grupo_t"] == "triage_normal").sum()),
                 "tasa_exp_%": round(100 * d.loc[d["grupo_t"] == "triage_anormal", "y"].mean(), 2),
                 "tasa_ref_%": round(100 * d.loc[d["grupo_t"] == "triage_normal", "y"].mean(), 2),
                 "OR M2 (IC95%)": fmt_or(o, lo, hi), "p": fmt_p(p)})
    return rows


def alt_defs(c, sitio, coh):
    rows = []
    ln = (c["news2_last"] <= NORM) & (c["maxcomp_last"] < 2)
    pm = c["news2_prior_max"].fillna(-1)
    defs = {"previo NEWS2 >=5; ultimo <=1": (ln & (pm >= 5), ln & (pm < ABN)),
            "previo NEWS2 >=3; ultimo = 0": ((c["news2_last"] == 0) & (pm >= ABN), (c["news2_last"] == 0) & (pm < ABN))}
    if "o2_any" in c:
        o2 = c["o2_any"].astype(bool)
        defs["principal, sin oxigeno suplementario"] = (c["grupo"].eq("normalizacion") & ~o2, c["grupo"].eq("estable") & ~o2)
    for nombre, (e, r) in defs.items():
        d0 = c[e | r].assign(exp=np.where(e[e | r], "si", "no"))
        r = {"centro": sitio, "cohorte": coh, "definicion": nombre, "n_expuestos": int(e.sum()),
             "eventos_expuestos": int(c.loc[e, "y"].sum()), "n_referencia": int(r.sum())}
        try:
            m, d = fit(d0, f"y ~ C(exp, Treatment('no')) + {COV['M2']}", NEEDED["M2"])
            o, lo, hi, p = term(m, "C(exp, Treatment('no'))[T.si]")
            r.update({"OR M2 (IC95%)": fmt_or(o, lo, hi), "p": fmt_p(p)})
        except Exception as ex:
            r.update({"OR M2 (IC95%)": f"no estimable ({type(ex).__name__})"})
        rows.append(r)
    return rows


def fenotipos(c, sitio, coh):
    rows = []
    s = c[c["grupo"] == "normalizacion"].copy()
    for clave, (var, cond, etiqueta) in FENOTIPOS.items():
        if var not in s or s[var].notna().sum() == 0:
            continue
        s["f"] = np.where(s[var].isna(), np.nan, np.where(cond(s[var]), 1, 0))
        dd = s.dropna(subset=["f"])
        n1, e1 = int((dd["f"] == 1).sum()), int(dd.loc[dd["f"] == 1, "y"].sum())
        n0, e0 = int((dd["f"] == 0).sum()), int(dd.loc[dd["f"] == 0, "y"].sum())
        r = {"centro": sitio, "cohorte": coh, "fenotipo": etiqueta, "n_con": n1, "eventos_con": e1,
             "tasa_con_%": round(100 * e1 / max(n1, 1), 2), "n_sin": n0, "tasa_sin_%": round(100 * e0 / max(n0, 1), 2),
             "faltantes": int(s["f"].isna().sum())}
        if e1 >= 10 and e0 >= 10:
            try:
                m, d = fit(dd, f"y ~ f + {COV['M2']}", NEEDED["M2"] + ["f"])
                o, lo, hi, p = term(m, "f")
                r.update({"OR M2 (IC95%)": fmt_or(o, lo, hi), "p": fmt_p(p)})
            except Exception as e:
                r.update({"OR M2 (IC95%)": f"no estimable ({type(e).__name__})"})
        else:
            r.update({"OR M2 (IC95%)": "muy pocos eventos"})
        rows.append(r)
    return rows


def missing_report(c, sitio, coh):
    cols = ["edad10", "sexo", "esi", "motivo", "llegada", "los_h", "h_ult",
            "sbp_min", "o2sat_min", "temperature_max", "temperature_min", "heartrate_max", "resprate_max"]
    return [{"centro": sitio, "cohorte": coh, "variable": k, "faltantes": int(c[k].isna().sum()),
             "pct": round(100 * c[k].isna().mean(), 2)} for k in cols if k in c]


# ----------------------------------------------------------------------------
# MC-MED: minimos y maximos por visita desde los sets (para fenotipos)
# ----------------------------------------------------------------------------
def mc_extremes(sets):
    g = sets.groupby("stay_id")
    ex = pd.DataFrame({"sbp_min": g["sbp"].min(), "o2sat_min": g["o2sat"].min(),
                       "temperature_max": g["temperature"].max(), "temperature_min": g["temperature"].min(),
                       "heartrate_max": g["heartrate"].max(), "resprate_max": g["resprate"].max()})
    return ex.reset_index()


# ----------------------------------------------------------------------------
# MC-MED: reconstruccion alternativa de sets (sensibilidad)
# ----------------------------------------------------------------------------
YEAR_RE = re.compile(r"\b(2[1-4]\d\d)\b")
VITALS = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature"]
PLAUS = {"heartrate": (20, 250), "resprate": (4, 60), "o2sat": (50, 100), "sbp": (50, 300), "dbp": (20, 200), "temperature": (30, 43)}
MMAP = {"HR": "heartrate", "RR": "resprate", "SpO2": "o2sat", "SBP": "sbp", "DBP": "dbp", "Temp": "temperature"}


def fix_time(s):
    s = s.astype("string")
    f = s.str.replace(YEAR_RE, lambda m: str(int(m.group(1)) - 400), regex=True)
    t = pd.to_datetime(f, format="ISO8601", errors="coerce")
    if t.notna().mean() < 0.99 * s.notna().mean():
        t = t.fillna(pd.to_datetime(f, format="mixed", errors="coerce"))
    return t


def score(x, bins, pts):
    out = pd.Series(np.nan, index=x.index); ok = x.notna()
    out[ok] = np.array(pts)[np.searchsorted(np.array(bins), x[ok].values, side="left")]
    return out


def news2(df):
    df = df.copy()
    for v in VITALS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    df["temperature"] = np.where(df["temperature"] > 50, (df["temperature"] - 32) * 5 / 9, df["temperature"])
    for v, (lo, hi) in PLAUS.items():
        df.loc[(df[v] < lo) | (df[v] > hi), v] = np.nan
    comps = {"s_rr": score(df["resprate"], [8, 11, 20, 24], [3, 1, 0, 2, 3]),
             "s_spo2": score(df["o2sat"], [91, 93, 95], [3, 2, 1, 0]),
             "s_sbp": score(df["sbp"], [90, 100, 110, 219], [3, 2, 1, 0, 3]),
             "s_hr": score(df["heartrate"], [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3]),
             "s_temp": score(df["temperature"].round(1), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])}
    S = pd.DataFrame(comps)
    df["n_params"], df["news2"], df["max_comp"] = S.notna().sum(1), S.sum(1, min_count=1), S.max(1)
    return df


def exposure_from_sets(long):
    long = long[long["n_params"] >= 4].sort_values(["stay_id", "charttime"])
    g = long.groupby("stay_id")
    last = g.tail(1).set_index("stay_id")
    long = long.assign(is_last=~long.duplicated("stay_id", keep="last"))
    prior = long[~long["is_last"]]
    pab = prior.assign(a=(prior["news2"] >= ABN) | (prior["max_comp"] >= 3)).groupby("stay_id")["a"].any()
    f = pd.DataFrame(index=last.index)
    f["n_sets"] = g.size()
    f["prior_abnormal"] = pab.reindex(f.index).fillna(False).astype(bool)
    f["last_normal"] = (last["news2"] <= NORM) & (last["max_comp"] < 2)
    f["apparent_normalization"] = f["prior_abnormal"] & f["last_normal"]
    f["stable_throughout"] = ~f["prior_abnormal"] & f["last_normal"]
    return f[f["n_sets"] >= 2].reset_index()


def mc_recon_sensitivity(cohorts):
    print("  Releyendo controles de enfermeria de MC-MED...")
    v = pd.read_csv(os.path.join(MC, "visits.csv"), low_memory=False, usecols=[
        "CSN", "Arrival_time", "Departure_time", "Triage_Temp", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP"])
    v["intime"], v["outtime"] = fix_time(v["Arrival_time"]), fix_time(v["Departure_time"])
    parts = []
    for ch in pd.read_csv(os.path.join(MC, "numerics.csv"), chunksize=5_000_000, usecols=["CSN", "Source", "Measure", "Value", "Time"]):
        parts.append(ch[(ch["Source"] == "Chart") & ch["Measure"].isin(MMAP)][["CSN", "Measure", "Value", "Time"]])
    ch = pd.concat(parts, ignore_index=True)
    ch["Time"] = fix_time(ch["Time"]); ch["Value"] = pd.to_numeric(ch["Value"], errors="coerce")
    ch = ch.dropna(subset=["Time", "Value"]).rename(columns={"CSN": "stay_id"}).sort_values(["stay_id", "Time"])
    ch["var"] = ch["Measure"].map(MMAP)
    tri = v.rename(columns={"CSN": "stay_id", "intime": "charttime", "Triage_Temp": "temperature", "Triage_HR": "heartrate",
                            "Triage_RR": "resprate", "Triage_SpO2": "o2sat", "Triage_SBP": "sbp", "Triage_DBP": "dbp"})
    rows = []
    for nombre, gap, locf in (("sin relleno, agrupacion 10 min", 10, 0), ("agrupacion estricta 5 min, relleno 60 min", 5, 60),
                              ("sin relleno, agrupacion 5 min", 5, 0)):
        x = ch.copy()
        new = (x["stay_id"] != x["stay_id"].shift()) | (x["Time"] - x["Time"].shift() > pd.Timedelta(minutes=gap))
        x["set_id"] = new.cumsum()
        sets = x.pivot_table(index=["stay_id", "set_id"], columns="var", values="Value", aggfunc="last").reset_index()
        sets = sets.merge(x.groupby("set_id")["Time"].max().rename("charttime"), on="set_id")
        for c_ in VITALS:
            if c_ not in sets:
                sets[c_] = np.nan
        long = pd.concat([tri[["stay_id", "charttime"] + VITALS], sets[["stay_id", "charttime"] + VITALS]], ignore_index=True)
        long = long.merge(v.rename(columns={"CSN": "stay_id"})[["stay_id", "intime", "outtime"]], on="stay_id")
        long = long[(long["charttime"] >= long["intime"] - pd.Timedelta(minutes=30)) & (long["charttime"] <= long["outtime"])]
        long = long.sort_values(["stay_id", "charttime"])
        if locf:
            for c_ in VITALS:
                lt = long["charttime"].where(long[c_].notna()).groupby(long["stay_id"]).ffill()
                fill = long.groupby("stay_id")[c_].ffill()
                long[c_] = long[c_].fillna(fill.where((long["charttime"] - lt) <= pd.Timedelta(minutes=locf)))
        long = news2(long)
        ex = exposure_from_sets(long)
        for coh, base in cohorts.items():
            b = base.drop(columns=["apparent_normalization", "stable_throughout", "last_normal", "prior_abnormal", "n_sets"],
                          errors="ignore").merge(ex, on="stay_id", how="inner")
            b = prep(b)
            m, d = fit(b, f"y ~ C(grupo, Treatment('estable')) + {COV['M2']}", NEEDED["M2"])
            o, lo, hi, p = term(m, "C(grupo, Treatment('estable'))[T.normalizacion]")
            rows.append({"centro": "MC-MED", "cohorte": coh, "reconstruccion": nombre,
                         "sets_validos_%": round(100 * (long["n_params"] >= 4).mean(), 1),
                         "n_cohorte": len(b), "normalizacion_%": round(100 * b["apparent_normalization"].mean(), 1),
                         "eventos_expuestos": int(b.loc[b["grupo"] == "normalizacion", "y"].sum()),
                         "OR M2 (IC95%)": fmt_or(o, lo, hi), "p": fmt_p(p)})
        print(f"    {nombre} listo")
    return rows


# ----------------------------------------------------------------------------
# Ejecucion
# ----------------------------------------------------------------------------
print("Cargando cohortes...")
mi = {"A": load_any(S_MI, "cohorte_A"), "B": load_any(S_MI, "cohorte_B")}
mc = {"A": load_any(S_MC, "mcmed_cohorte_A"), "B*": load_any(S_MC, "mcmed_cohorte_B")}
sets_mc = pd.read_csv(os.path.join(S_MC, "mcmed_sets.csv.gz"))
ex_mc = mc_extremes(sets_mc)
for k in mc:
    mc[k] = mc[k].merge(ex_mc, on="stay_id", how="left")

R = {k: [] for k in ("contrastes", "frecuencia", "definiciones", "fenotipos", "faltantes")}
for sitio, cohs in (("MIMIC-IV-ED", mi), ("MC-MED", mc)):
    for coh, c in cohs.items():
        print(f"  {sitio} cohorte {coh}...")
        c = prep(c)
        R["contrastes"] += contrasts(c, sitio, coh)
        R["frecuencia"] += freq_bias(c, sitio, coh)
        R["definiciones"] += alt_defs(c, sitio, coh)
        R["fenotipos"] += fenotipos(c, sitio, coh)
        R["faltantes"] += missing_report(c, sitio, coh)
if RUN_MC_RECON:
    R["reconstruccion_mcmed"] = mc_recon_sensitivity({"A": mc["A"], "B*": mc["B*"]})

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)
titulos = {"contrastes": "1-2. Asociacion principal (M2) y sensibilidades; contraste directo vs ultimo set anormal",
           "frecuencia": "3. Sesgo por frecuencia de medicion",
           "definiciones": "4. Definiciones alternativas (ajustadas, M2)",
           "reconstruccion_mcmed": "5. MC-MED: reconstruccion alternativa de los sets (M2)",
           "fenotipos": "6. Fenotipos preespecificados dentro de normalizacion aparente (M2; asociativo)",
           "faltantes": "Datos faltantes por variable"}
for k, t in titulos.items():
    if k in R and R[k]:
        df = pd.DataFrame(R[k])
        print(f"\n=== {t} ===")
        print(df.to_string(index=False))
        df.to_csv(os.path.join(OUT7, f"fase7a_{k}.csv"), index=False)
print(f"\nGuardado en {OUT7}")
