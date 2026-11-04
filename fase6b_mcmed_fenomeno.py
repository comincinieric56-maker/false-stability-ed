# ============================================================================
# FALSE-STABILITY - Fase 6b: validacion externa del fenomeno en MC-MED
#   Replica las fases 2 y 2b de MIMIC con la MISMA definicion (sin reajustar).
#   Sets de signos vitales = controles registrados por enfermeria (Source=Chart)
#   + triage. Cohorte A: alta -> reconsulta con hospitalizacion <=72 h.
#   Cohorte B*: sala/observacion -> muerte intrahospitalaria o alta a hospicio
#   (sustituto: MC-MED no registra traslados de sala a UCI).
# Requiere Drive montado. CPU. ~10-15 minutos. Salida solo agregada.
# ============================================================================
import os, re, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
MC = globals().get("dest", "/content/drive/MyDrive/ARTICULOS MIOS/False Stability/mcmed")
OUT_MC = os.path.join(os.path.dirname(MC), "salida_mcmed")
os.makedirs(OUT_MC, exist_ok=True)

# --- Misma definicion que MIMIC (congelada) ---
VITALS = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature"]
PLAUSIBLE = {"heartrate": (20, 250), "resprate": (4, 60), "o2sat": (50, 100),
             "sbp": (50, 300), "dbp": (20, 200), "temperature": (30, 43)}
ABNORMAL_NEWS, NORMAL_NEWS, MIN_PARAMS = 3, 1, 4
REVISIT_H = 72
SET_GAP_MIN = 10      # registros de enfermeria a <=10 min se agrupan en un mismo set
LOCF_MIN = 60         # un parametro faltante se completa con su ultimo valor de <=60 min
MEASURE_MAP = {"HR": "heartrate", "RR": "resprate", "SpO2": "o2sat", "SBP": "sbp",
               "DBP": "dbp", "Temp": "temperature", "LPM_O2": "o2_lpm"}


# ----------------------------------------------------------------------------
# Fechas: MC-MED desplaza fechas hasta ~2300; pandas llega a 2262.
# Se restan 400 anos (conserva intervalos y anos bisiestos).
# ----------------------------------------------------------------------------
YEAR_RE = re.compile(r"\b(2[1-4]\d\d)\b")


def fix_time(s):
    s = s.astype("string")
    fixed = s.str.replace(YEAR_RE, lambda m: str(int(m.group(1)) - 400), regex=True)
    t = pd.to_datetime(fixed, format="ISO8601", errors="coerce")
    if t.notna().mean() < 0.99 * s.notna().mean():
        t2 = pd.to_datetime(fixed, format="mixed", errors="coerce")
        t = t.fillna(t2)
    return t


# ----------------------------------------------------------------------------
# NEWS2 modificado (identico a MIMIC)
# ----------------------------------------------------------------------------
def score(x, bins, pts):
    out = pd.Series(np.nan, index=x.index)
    ok = x.notna()
    out[ok] = np.array(pts)[np.searchsorted(np.array(bins), x[ok].values, side="left")]
    return out


def clean_vitals(df):
    df = df.copy()
    for v in VITALS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    t = df["temperature"]
    df["temperature"] = np.where(t > 50, (t - 32) * 5 / 9, t)
    for v, (lo, hi) in PLAUSIBLE.items():
        df.loc[(df[v] < lo) | (df[v] > hi), v] = np.nan
    return df


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
    # Sensibilidad: NEWS2 con oxigeno suplementario (+2 si recibe O2)
    df["news2_o2"] = df["news2"] + np.where(df.get("o2_lpm", pd.Series(0, index=df.index)).fillna(0) > 0, 2, 0)
    return df


def trajectory_features(long):
    g = long.groupby("stay_id", sort=False)
    last, first = g.tail(1).set_index("stay_id"), g.head(1).set_index("stay_id")
    long = long.copy()
    long["is_last"] = ~long.duplicated("stay_id", keep="last")
    prior = long[~long["is_last"]]
    prior_abn = (prior["news2"] >= ABNORMAL_NEWS) | (prior["max_comp"] >= 3)
    f = pd.DataFrame(index=last.index)
    f["n_sets"] = g.size()
    f["news2_first"], f["news2_last"] = first["news2"], last["news2"]
    f["news2_max"] = g["news2"].max()
    f["news2_prior_max"] = prior.groupby("stay_id")["news2"].max()
    f["maxcomp_last"] = last["max_comp"]
    f["prior_abnormal"] = prior.assign(a=prior_abn).groupby("stay_id")["a"].any().reindex(f.index).fillna(False).astype(bool)
    f["last_normal"] = (last["news2"] <= NORMAL_NEWS) & (last["max_comp"] < 2)
    f["last_set_time"] = last["charttime"]
    f["o2_any"] = g["o2_lpm"].apply(lambda s: (s.fillna(0) > 0).any())
    f["apparent_normalization"] = f["prior_abnormal"] & f["last_normal"]
    f["stable_throughout"] = (~f["prior_abnormal"]) & f["last_normal"]
    return f.reset_index()


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
print("Leyendo visits...")
v = pd.read_csv(os.path.join(MC, "visits.csv"), low_memory=False)
for col in ["Arrival_time", "Departure_time"]:
    v[col] = fix_time(v[col])
print(f"  fechas legibles: llegada {v['Arrival_time'].notna().mean():.1%}, salida {v['Departure_time'].notna().mean():.1%}")
v = v.rename(columns={"CSN": "stay_id", "MRN": "subject_id", "Arrival_time": "intime", "Departure_time": "outtime"})

print("Leyendo controles de enfermeria (numerics, Source=Chart)...")
parts = []
for ch in pd.read_csv(os.path.join(MC, "numerics.csv"), chunksize=5_000_000,
                      usecols=["CSN", "Source", "Measure", "Value", "Time"]):
    ch = ch[(ch["Source"] == "Chart") & ch["Measure"].isin(MEASURE_MAP)]
    parts.append(ch[["CSN", "Measure", "Value", "Time"]])
chart = pd.concat(parts, ignore_index=True)
chart["Time"] = fix_time(chart["Time"])
chart["Value"] = pd.to_numeric(chart["Value"], errors="coerce")
print(f"  {len(chart):,} registros | fechas legibles {chart['Time'].notna().mean():.1%} | visitas {chart['CSN'].nunique():,}")
chart = chart.dropna(subset=["Time", "Value"]).rename(columns={"CSN": "stay_id"})

# Agrupar registros cercanos en sets
chart = chart.sort_values(["stay_id", "Time"])
new = (chart["stay_id"] != chart["stay_id"].shift()) | \
      (chart["Time"] - chart["Time"].shift() > pd.Timedelta(minutes=SET_GAP_MIN))
chart["set_id"] = new.cumsum()
chart["var"] = chart["Measure"].map(MEASURE_MAP)
sets = chart.pivot_table(index=["stay_id", "set_id"], columns="var", values="Value", aggfunc="last").reset_index()
sets = sets.merge(chart.groupby("set_id")["Time"].max().rename("charttime"), on="set_id")
for c in VITALS + ["o2_lpm"]:
    if c not in sets:
        sets[c] = np.nan
sets["source"] = "chart"

tri = v[["stay_id", "intime", "Triage_Temp", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP"]].rename(
    columns={"intime": "charttime", "Triage_Temp": "temperature", "Triage_HR": "heartrate", "Triage_RR": "resprate",
             "Triage_SpO2": "o2sat", "Triage_SBP": "sbp", "Triage_DBP": "dbp"})
tri["o2_lpm"], tri["source"] = np.nan, "triage"

cols = ["stay_id", "charttime", "source"] + VITALS + ["o2_lpm"]
long = pd.concat([tri[cols], sets[cols]], ignore_index=True)
long = long.merge(v[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
long = long[(long["charttime"] >= long["intime"] - pd.Timedelta(minutes=30)) & (long["charttime"] <= long["outtime"])]
long = clean_vitals(long).sort_values(["stay_id", "charttime"])

# Diagnostico antes de completar
pre_valid = (long[["heartrate", "resprate", "o2sat", "sbp", "temperature"]].notna().sum(axis=1) >= MIN_PARAMS).mean()

# Ultimo valor conocido de <=60 min para parametros faltantes (dentro de la visita)
for c in VITALS:
    last_t = long["charttime"].where(long[c].notna())
    last_t = last_t.groupby(long["stay_id"]).ffill()
    filled = long.groupby("stay_id")[c].ffill()
    ok = (long["charttime"] - last_t) <= pd.Timedelta(minutes=LOCF_MIN)
    long[c] = long[c].fillna(filled.where(ok))
long = news2_modified(long)
print(f"  sets validos (>=4 parametros): antes de completar {pre_valid:.1%}, despues {long['valid_set'].mean():.1%}")
long = long[long["valid_set"]]

feats = trajectory_features(long)
feats = feats.merge(v[["stay_id", "outtime"]], on="stay_id")
feats["hours_last_set_to_exit"] = (feats["outtime"] - feats["last_set_time"]).dt.total_seconds() / 3600
feats = feats.drop(columns="outtime")

# ----------------------------------------------------------------------------
# Cohortes
# ----------------------------------------------------------------------------
v["age"] = pd.to_numeric(v["Age"], errors="coerce")
v["dc"] = v["DC_dispo"].fillna("").astype(str)
v["acuity"] = pd.to_numeric(v["Triage_acuity"].astype(str).str[0], errors="coerce")
v = v.assign(gender=v["Gender"], arrival_transport=v["Means_of_arrival"], chiefcomplaint=v["CC"])
v["ed_los_h"] = pd.to_numeric(v["ED_LOS"], errors="coerce")

# Diagnostico: Hours_to_next_visit se mide desde la llegada o desde la salida?
vv = v.sort_values(["subject_id", "intime"])
nxt_in = vv.groupby("subject_id")["intime"].shift(-1)
cmp = pd.DataFrame({"prov": pd.to_numeric(vv["Hours_to_next_visit"], errors="coerce"),
                    "desde_llegada": (nxt_in - vv["intime"]).dt.total_seconds() / 3600,
                    "desde_salida": (nxt_in - vv["outtime"]).dt.total_seconds() / 3600}).dropna()
print("\nHours_to_next_visit (con siguiente visita en la base, n=%d): |dif| mediana desde llegada %.2f h, desde salida %.2f h"
      % (len(cmp), (cmp["prov"] - cmp["desde_llegada"]).abs().median(), (cmp["prov"] - cmp["desde_salida"]).abs().median()))

flow = []
nh = pd.to_numeric(v["Hours_to_next_visit"], errors="coerce")
v["revisit_72h_admit"] = (nh <= REVISIT_H) & v["Dispo_class_next_visit"].isin(["Inpatient", "Observation", "ICU"])

a = v[v["ED_dispo"] == "Discharge"]
flow.append(("A", "Alta desde urgencia", len(a)))
a = a[~a["dc"].str.contains("Against Medical Advice|Left Without|Expired", case=False)]
flow.append(("A", "Sin fuga/alta voluntaria/muerte", len(a)))
a = a.merge(feats, on="stay_id", how="inner")
a = a[a["n_sets"] >= 2]
flow.append(("A", ">= 2 sets validos", len(a)))
a["outcome"] = a["revisit_72h_admit"]

b = v[v["ED_dispo"].isin(["Inpatient", "Observation"])]
flow.append(("B*", "Hospitalizacion en sala u observacion", len(b)))
b = b.merge(feats, on="stay_id", how="inner")
b = b[b["n_sets"] >= 2]
flow.append(("B*", ">= 2 sets validos", len(b)))
b["outcome"] = b["dc"].str.contains("Expired|Hospice", case=False)
hl = pd.to_numeric(b["Hosp_LOS"], errors="coerce")
print(f"Cohorte B*: muertes {b['dc'].str.contains('Expired', case=False).sum():,} "
      f"(con estadia <=2 dias: {(b['dc'].str.contains('Expired', case=False) & (hl <= 2)).sum():,}), "
      f"altas a hospicio {b['dc'].str.contains('Hospice', case=False).sum():,}")

for c in (a, b):
    c["sets_per_hour"] = c["n_sets"] / c["ed_los_h"].clip(lower=0.25)
a.to_csv(os.path.join(OUT_MC, "mcmed_cohorte_A.csv.gz"), index=False)
b.to_csv(os.path.join(OUT_MC, "mcmed_cohorte_B.csv.gz"), index=False)
long[["stay_id", "charttime", "source"] + VITALS + ["o2_lpm", "s_rr", "s_spo2", "s_sbp", "s_hr", "s_temp", "news2"]] \
    .to_csv(os.path.join(OUT_MC, "mcmed_sets.csv.gz"), index=False)


# ----------------------------------------------------------------------------
# Analisis (mismas tablas que MIMIC)
# ----------------------------------------------------------------------------
def rr_ci(e1, n1, e0, n0):
    rr = (e1 / n1) / (e0 / n0)
    se = np.sqrt(1 / e1 - 1 / n1 + 1 / e0 - 1 / n0)
    return f"{rr:.2f} ({rr * np.exp(-1.96 * se):.2f}-{rr * np.exp(1.96 * se):.2f})"


def summarize(c, name):
    rows = []
    for label, m in (("Toda la cohorte", pd.Series(True, index=c.index)), ("Normalizacion aparente", c["apparent_normalization"]),
                     ("Estable desde triage", c["stable_throughout"]), ("Ultimo set anormal", ~c["last_normal"])):
        s = c[m]
        rows.append({"cohorte": name, "grupo": label, "n": len(s), "pct": round(100 * len(s) / len(c), 1),
                     "eventos": int(s["outcome"].sum()), "tasa_pct": round(100 * s["outcome"].mean(), 2)})
    return pd.DataFrame(rows)


def prep_reg(c, top_cc=30):
    c = c.copy()
    c["grupo"] = np.select([c["apparent_normalization"], c["stable_throughout"]], ["normalizacion", "estable"], "ultimo_anormal")
    c["y"] = c["outcome"].astype(int)
    c["sexo"] = c["gender"].fillna("NA").astype(str)
    c["esi"] = c["acuity"].fillna(-1).astype(int).astype(str)
    c["llegada"] = c["arrival_transport"].fillna("NA").astype(str)
    cc = c["chiefcomplaint"].fillna("NA").astype(str).str.lower().str.strip()
    c["motivo"] = np.where(cc.isin(cc.value_counts().index[:top_cc]), cc, "otro")
    c["edad10"] = c["age"] / 10
    c["log_sets"] = np.log(c["n_sets"])
    c["los_h"] = c["ed_los_h"].clip(0, 72)
    c["h_ult"] = c["hours_last_set_to_exit"].clip(0, 48)
    return c


MODELOS = {"M1 edad+sexo+ESI": "y ~ C(grupo, Treatment('estable')) + edad10 + C(sexo) + C(esi)",
           "M2 + motivo+llegada": "y ~ C(grupo, Treatment('estable')) + edad10 + C(sexo) + C(esi) + C(motivo) + C(llegada)",
           "M3 + proceso": "y ~ C(grupo, Treatment('estable')) + edad10 + C(sexo) + C(esi) + C(motivo) + C(llegada) + log_sets + los_h + h_ult"}


def ajustado(c, name):
    rows = []
    d = c[["y", "grupo", "edad10", "sexo", "esi", "motivo", "llegada", "log_sets", "los_h", "h_ult", "subject_id"]].dropna()
    for mn, f in MODELOS.items():
        m = smf.glm(f, data=d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d["subject_id"].values})
        for g in ("normalizacion", "ultimo_anormal"):
            k = f"C(grupo, Treatment('estable'))[T.{g}]"
            if k in m.params:
                b_, se = m.params[k], m.bse[k]
                rows.append({"cohorte": name, "modelo": mn, "grupo vs estable": g, "OR": round(np.exp(b_), 2),
                             "IC95": f"{np.exp(b_ - 1.96 * se):.2f}-{np.exp(b_ + 1.96 * se):.2f}"})
    return pd.DataFrame(rows)


def sensibilidad(c, name):
    ln = (c["news2_last"] <= 1) & (c["maxcomp_last"] < 2)
    pm = c["news2_prior_max"].fillna(-1)
    defs = {"Principal": (c["apparent_normalization"], c["stable_throughout"]),
            "Previo NEWS2 >=5; ultimo <=1": (ln & (pm >= 5), ln & (pm < 3)),
            "Previo NEWS2 >=3; ultimo = 0": ((c["news2_last"] == 0) & (pm >= 3), (c["news2_last"] == 0) & (pm < 3)),
            "Principal, sin O2 suplementario": (c["apparent_normalization"] & ~c["o2_any"], c["stable_throughout"] & ~c["o2_any"])}
    rows = []
    for dn, (e, r) in defs.items():
        e1, n1, e0, n0 = int(c.loc[e, "outcome"].sum()), int(e.sum()), int(c.loc[r, "outcome"].sum()), int(r.sum())
        rows.append({"cohorte": name, "definicion": dn, "n_expuestos": n1, "eventos": e1,
                     "tasa_exp_pct": round(100 * e1 / max(n1, 1), 2), "tasa_estables_pct": round(100 * e0 / max(n0, 1), 2),
                     "RR (IC95)": rr_ci(e1, n1, e0, n0) if e1 and e0 else "NA"})
    return pd.DataFrame(rows)


def tabla1(c, name):
    g = c.groupby("grupo")
    t = pd.DataFrame({"n": g.size(), "edad_mediana": g["age"].median(), "pct_ESI_1_2": g["esi"].apply(lambda s: round(100 * s.isin(["1", "2"]).mean(), 1)),
                      "sets_mediana": g["n_sets"].median(), "news2_max_mediana": g["news2_max"].median(),
                      "estadia_h_mediana": g["ed_los_h"].median().round(1), "h_ultimo_set_mediana": g["hours_last_set_to_exit"].median().round(2),
                      "tasa_pct": (100 * g["y"].mean()).round(2)})
    t.insert(0, "cohorte", name)
    return t.reset_index()


def por_ausencia(c, name):
    c = c.copy()
    c["h_cat"] = pd.cut(c["hours_last_set_to_exit"], [-0.01, 1, 3, 6, np.inf], labels=["<=1 h", "1-3 h", "3-6 h", ">6 h"])
    t = c[c["grupo"] != "ultimo_anormal"].groupby(["h_cat", "grupo"], observed=True)["y"].agg(
        n="size", eventos="sum", tasa_pct=lambda s: round(100 * s.mean(), 2)).reset_index()
    t.insert(0, "cohorte", name)
    return t


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
names = (("A: reconsulta 72h", a), ("B*: muerte/hospicio", b))
res = {"flujo": pd.DataFrame(flow, columns=["cohorte", "paso", "n"]),
       "fenomeno": pd.concat([summarize(c, n) for n, c in names])}
regs = {n: prep_reg(c) for n, c in names}
res["tabla1"] = pd.concat([tabla1(c, n) for n, c in regs.items()])
res["ajustado"] = pd.concat([ajustado(c, n) for n, c in regs.items()])
res["sensibilidad"] = pd.concat([sensibilidad(c, n) for n, c in names])
res["ausencia"] = pd.concat([por_ausencia(c, n) for n, c in regs.items()])

for titulo, k in (("Flujo de seleccion", "flujo"), ("Fenomeno de falsa estabilidad", "fenomeno"), ("Tabla 1 por grupo", "tabla1"),
                  ("OR ajustados (referencia: estables, SE por paciente)", "ajustado"), ("Sensibilidad de la definicion", "sensibilidad"),
                  ("Antiguedad del ultimo control", "ausencia")):
    print(f"\n=== {titulo} ===")
    print(res[k].to_string(index=False))
    res[k].to_csv(os.path.join(OUT_MC, f"fase6b_{k}.csv"), index=False)
print(f"\nArchivos en {OUT_MC}")
