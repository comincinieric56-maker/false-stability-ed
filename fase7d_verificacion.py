# ============================================================================
# FALSE-STABILITY - Fase 7d: verificacion de los hallazgos de la auditoria
# Recalcula, sobre los datos originales, las cuatro situaciones senaladas y
# re-estima el modelo principal (M2) con ambos contrastes. Independiente.
#
#   V1  Reconsulta 72 h: CUALQUIER visita en la ventana que termine en ingreso
#       (el codigo original miraba solo la PRIMERA visita siguiente).
#   V2  Definicion "solo triage" usando el registro de TRIAGE real
#       (el original usaba el primer set valido).
#   V3  Sets completos: solo sets con los 5 parametros (el original acepta 4/5,
#       y un parametro faltante cuenta como 0 puntos).
#   V4  MC-MED: ventanas fijas de 10 min (sin encadenar registros) y
#       distribucion de la duracion de los sets originales.
#
# Requiere en Drive: mimic-iv-ed-2.2.zip, salida/, salida_mcmed/, mcmed/.
# CPU, ~20-30 min (V4 relee numerics de MC-MED). Salida solo agregada.
# ============================================================================
import os, re, zipfile, warnings
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
ED_ZIP = os.path.join(FOLDER, "mimic-iv-ed-2.2.zip")
OUT7 = os.path.join(FOLDER, "salida_fase7")
os.makedirs(OUT7, exist_ok=True)
RUN_V4 = True
# Para ahorrar memoria: corre una parte por vez y REINICIA el entorno entre ellas
#   PARTE = "mimic"  -> solo MIMIC-IV-ED       PARTE = "mcmed" -> solo MC-MED
#   PARTE = "todo"   -> ambas (requiere entorno con mucha RAM)
PARTE = globals().get("PARTE", "mimic")
import gc
COLS = ["stay_id", "subject_id", "outcome", "apparent_normalization", "stable_throughout", "last_normal",
        "prior_abnormal", "n_sets", "news2_first", "news2_last", "maxcomp_last", "gender", "acuity",
        "arrival_transport", "chiefcomplaint", "age"]

ABN, NORM, MIN_PARAMS = 3, 1, 4
VITALS = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature"]
COMPS = ["s_rr", "s_spo2", "s_sbp", "s_hr", "s_temp"]
PLAUS = {"heartrate": (20, 250), "resprate": (4, 60), "o2sat": (50, 100), "sbp": (50, 300), "dbp": (20, 200), "temperature": (30, 43)}
EXPO = ["apparent_normalization", "stable_throughout", "last_normal", "prior_abnormal", "n_sets", "news2_first", "news2_last", "maxcomp_last"]


# ----------------------------------------------------------------------------
# Utilidades comunes (identicas a las fases anteriores)
# ----------------------------------------------------------------------------
def load_any(folder, name):
    for ext, rd in ((".parquet", lambda p: (lambda d: d[[c for c in COLS if c in d]])(pd.read_parquet(p))),
                    (".csv.gz", lambda p: pd.read_csv(p, low_memory=False, usecols=lambda c: c in COLS))):
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return rd(p)
    raise FileNotFoundError(name)


def read_from_zip(table, **kw):
    with zipfile.ZipFile(ED_ZIP) as z:
        n = [x for x in z.namelist() if os.path.basename(x).startswith(table + ".csv")][0]
        with z.open(n) as f:
            return pd.read_csv(f, compression="gzip" if n.endswith(".gz") else None, **kw)


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
    df["s_rr"] = score(df["resprate"], [8, 11, 20, 24], [3, 1, 0, 2, 3])
    df["s_spo2"] = score(df["o2sat"], [91, 93, 95], [3, 2, 1, 0])
    df["s_sbp"] = score(df["sbp"], [90, 100, 110, 219], [3, 2, 1, 0, 3])
    df["s_hr"] = score(df["heartrate"], [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    df["s_temp"] = score(df["temperature"].round(1), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    return add_scores(df)


def add_scores(df):
    df["n_params"] = df[COMPS].notna().sum(axis=1)
    df["news2"] = df[COMPS].sum(axis=1, min_count=1)
    df["max_comp"] = df[COMPS].max(axis=1)
    return df


def exposure(long, min_params=MIN_PARAMS):
    """Grupos a partir de una tabla larga de sets (misma regla que la fase 2)."""
    long = long[long["n_params"] >= min_params].sort_values(["stay_id", "charttime"])
    g = long.groupby("stay_id")
    last, first = g.tail(1).set_index("stay_id"), g.head(1).set_index("stay_id")
    long = long.assign(is_last=~long.duplicated("stay_id", keep="last"))
    prior = long[~long["is_last"]]
    pab = prior.assign(a=(prior["news2"] >= ABN) | (prior["max_comp"] >= 3)).groupby("stay_id")["a"].any()
    f = pd.DataFrame(index=last.index)
    f["n_sets"] = g.size()
    f["news2_first"], f["news2_last"], f["maxcomp_last"] = first["news2"], last["news2"], last["max_comp"]
    f["prior_abnormal"] = pab.reindex(f.index).fillna(False).astype(bool)
    f["last_normal"] = (last["news2"] <= NORM) & (last["max_comp"] < 2)
    f["apparent_normalization"] = f["prior_abnormal"] & f["last_normal"]
    f["stable_throughout"] = ~f["prior_abnormal"] & f["last_normal"]
    return f[f["n_sets"] >= 2].reset_index()


def prep(c, top_cc=30):
    c = c.copy()
    c["grupo"] = np.select([c["apparent_normalization"].astype(bool), c["stable_throughout"].astype(bool)],
                           ["normalizacion", "estable"], "ultimo_anormal")
    c["y"] = c["outcome"].astype(bool).astype(int)
    c["sexo"] = c["gender"].where(c["gender"].isin(["F", "M"]))
    c["esi"] = pd.to_numeric(c["acuity"], errors="coerce").round().astype("Int64").astype(str).replace("<NA>", np.nan)
    arr = c["arrival_transport"].fillna("").astype(str).str.upper()
    c["llegada"] = np.select([arr.str.contains("WALK|SELF"), arr.str.contains("AMBUL|EMS|HELI")], ["propio", "ems"], "otro")
    cc = c["chiefcomplaint"].fillna("NA").astype(str).str.lower().str.strip()
    c["motivo"] = np.where(cc.isin(cc.value_counts().index[:top_cc]), cc, "otro")
    c["edad10"] = pd.to_numeric(c["age"], errors="coerce") / 10
    return c


M2 = "edad10 + C(sexo) + C(esi) + C(motivo) + C(llegada)"
NEED = ["edad10", "sexo", "esi", "motivo", "llegada", "y", "subject_id"]


def or_m2(c, ref):
    d = c.dropna(subset=NEED)
    m = smf.glm(f"y ~ C(grupo, Treatment('{ref}')) + {M2}", data=d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d["subject_id"].values})
    k = f"C(grupo, Treatment('{ref}'))[T.normalizacion]"
    b, se = m.params[k], m.bse[k]
    return f"{np.exp(b):.2f} ({np.exp(b - 1.96 * se):.2f}-{np.exp(b + 1.96 * se):.2f})", len(d), int(d["y"].sum())


def resumen(c, sitio, coh, variante, extra=None):
    c = prep(c)
    a, n, ev = or_m2(c, "estable")
    b, _, _ = or_m2(c, "ultimo_anormal")
    r = {"centro": sitio, "cohorte": coh, "variante": variante, "n_analizado": n, "eventos": ev,
         "normalizacion_%": round(100 * c["apparent_normalization"].mean(), 1),
         "tasa_normalizacion_%": round(100 * c.loc[c["grupo"] == "normalizacion", "y"].mean(), 2),
         "OR vs estable": a, "OR vs ultimo anormal": b}
    r.update(extra or {})
    print(f"    {variante}: OR vs estable {a} | vs ultimo anormal {b}")
    return r


def with_exposure(base, ex):
    return base.drop(columns=[x for x in EXPO if x in base], errors="ignore").merge(ex, on="stay_id", how="inner")


def triage_only(base, long, tri_source):
    """Ultimo set normal; triage REAL anormal (NEWS2 >= 3) frente a triage real normal."""
    t = long[(long["source"] == tri_source) & (long["n_params"] >= MIN_PARAMS)].drop_duplicates("stay_id")
    t = t.set_index("stay_id")["news2"].rename("news2_triage")
    c = base.merge(t, left_on="stay_id", right_index=True, how="inner")
    c = c[(c["news2_last"] <= NORM) & (c["maxcomp_last"] < 2)].copy()
    c["apparent_normalization"] = c["news2_triage"] >= ABN
    c["stable_throughout"] = ~c["apparent_normalization"]
    c = prep(c)
    a, n, ev = or_m2(c, "estable")
    return {"variante": "V2 solo triage (registro de triage real)", "n_analizado": n, "eventos": ev,
            "normalizacion_%": round(100 * c["apparent_normalization"].mean(), 1),
            "tasa_normalizacion_%": round(100 * c.loc[c["grupo"] == "normalizacion", "y"].mean(), 2),
            "OR vs estable": a, "OR vs ultimo anormal": "no aplica",
            "visitas_sin_triage_valido": int(len(base) - base["stay_id"].isin(t.index).sum())}


R = []
# ============================================================================
# MIMIC-IV-ED
# ============================================================================
if PARTE in ("mimic", "todo"):
  print("### MIMIC-IV-ED")
  ed = read_from_zip("edstays", usecols=["subject_id", "stay_id", "intime", "outtime", "hadm_id"], parse_dates=["intime", "outtime"])
  coh = {"A": load_any(S_MI, "cohorte_A"), "B": load_any(S_MI, "cohorte_B")}

  # V1 reconsulta: cualquier visita con ingreso en (salida, salida + 72 h]
  a = coh["A"]
  idx = a[["stay_id", "subject_id"]].merge(ed[["stay_id", "outtime"]], on="stay_id")
  nx = ed.loc[ed["hadm_id"].notna() & ed["subject_id"].isin(a["subject_id"]), ["subject_id", "intime", "hadm_id"]] \
      .rename(columns={"intime": "n_in", "hadm_id": "n_hadm"})   # solo visitas con ingreso: menos filas
  m = idx.merge(nx, on="subject_id")
  m = m[(m["n_in"] > m["outtime"]) & (m["n_in"] <= m["outtime"] + pd.Timedelta(hours=72))]
  any_adm = m.groupby("stay_id")["n_hadm"].apply(lambda s: s.notna().any())
  new_y = a["stay_id"].map(any_adm).fillna(False).astype(bool)
  old_y = a["outcome"].astype(bool)
  print(f"  V1: eventos originales {old_y.sum():,} -> corregidos {new_y.sum():,} (+{(new_y & ~old_y).sum():,}; perdidos {(old_y & ~new_y).sum():,})")
  R.append(resumen(a, "MIMIC-IV-ED", "A", "original"))
  R.append(resumen(a.assign(outcome=new_y), "MIMIC-IV-ED", "A", "V1 reconsulta: cualquier visita con ingreso",
                   {"eventos_agregados": int((new_y & ~old_y).sum())}))

  # Tabla larga de sets de MIMIC (misma construccion que la fase 2)
  print("  Reconstruyendo sets de MIMIC...")
  tri = read_from_zip("triage", usecols=["stay_id"] + VITALS)
  vs = read_from_zip("vitalsign", usecols=["stay_id", "charttime"] + VITALS, parse_dates=["charttime"])
  keep = set(pd.concat([coh["A"]["stay_id"], coh["B"]["stay_id"]]))
  vs, tri = vs[vs["stay_id"].isin(keep)], tri[tri["stay_id"].isin(keep)]
  tri = tri.merge(ed[["stay_id", "intime"]], on="stay_id").rename(columns={"intime": "charttime"}).assign(source="triage")
  vs = vs.assign(source="vitalsign")
  long = pd.concat([tri[["stay_id", "charttime", "source"] + VITALS], vs[["stay_id", "charttime", "source"] + VITALS]])
  long = long.merge(ed[["stay_id", "intime", "outtime"]], on="stay_id")
  long = long[(long["charttime"] >= long["intime"] - pd.Timedelta(minutes=30)) & (long["charttime"] <= long["outtime"])]
  del tri, vs; gc.collect()
  long = news2(long).drop(columns=["intime", "outtime", "dbp"], errors="ignore")
  for c_ in VITALS + COMPS + ["news2", "max_comp"]:
      if c_ in long:
          long[c_] = long[c_].astype("float32")
  gc.collect()
  ex5 = exposure(long, min_params=5)
  for k, base in coh.items():
      if k == "B":
          R.append(resumen(base, "MIMIC-IV-ED", "B", "original"))
      r = triage_only(base, long, "triage"); r.update({"centro": "MIMIC-IV-ED", "cohorte": k}); R.append(r)
      print(f"    {r['variante']}: OR {r['OR vs estable']} (sin triage valido: {r['visitas_sin_triage_valido']:,})")
      c5 = with_exposure(base, ex5)
      R.append(resumen(c5, "MIMIC-IV-ED", k, "V3 solo sets completos (5/5)", {"visitas_perdidas": int(len(base) - len(c5))}))
  del long, ed; gc.collect()

# ============================================================================
# MC-MED
# ============================================================================
if PARTE in ("mcmed", "todo"):
  print("\n### MC-MED")
  YEAR_RE = re.compile(r"\b(2[1-4]\d\d)\b")


  def fix_time(s):
      s = s.astype("string")
      f = s.str.replace(YEAR_RE, lambda m_: str(int(m_.group(1)) - 400), regex=True)
      t = pd.to_datetime(f, format="ISO8601", errors="coerce")
      if t.notna().mean() < 0.99 * s.notna().mean():
          t = t.fillna(pd.to_datetime(f, format="mixed", errors="coerce"))
      return t


  v = pd.read_csv(os.path.join(MC, "visits.csv"), low_memory=False,
                  usecols=["MRN", "CSN", "Arrival_time", "Departure_time", "ED_dispo"])
  v["intime"], v["outtime"] = fix_time(v["Arrival_time"]), fix_time(v["Departure_time"])
  mc = {"A": load_any(S_MC, "mcmed_cohorte_A"), "B*": load_any(S_MC, "mcmed_cohorte_B")}

  a = mc["A"]
  idx = a[["stay_id", "subject_id"]].merge(v[["CSN", "outtime"]].rename(columns={"CSN": "stay_id"}), on="stay_id")
  nx = v.loc[v["ED_dispo"].isin(["Inpatient", "Observation", "ICU"]) & v["MRN"].isin(a["subject_id"]), ["MRN", "intime", "ED_dispo"]] \
      .rename(columns={"MRN": "subject_id", "intime": "n_in"})
  m = idx.merge(nx, on="subject_id")
  m = m[(m["n_in"] > m["outtime"]) & (m["n_in"] <= m["outtime"] + pd.Timedelta(hours=72))]
  any_adm = m.groupby("stay_id")["ED_dispo"].apply(lambda s: s.isin(["Inpatient", "Observation", "ICU"]).any())
  new_y = a["stay_id"].map(any_adm).fillna(False).astype(bool)
  old_y = a["outcome"].astype(bool)
  print(f"  V1: eventos originales {old_y.sum():,} -> corregidos {new_y.sum():,} (+{(new_y & ~old_y).sum():,}; perdidos {(old_y & ~new_y).sum():,})")
  R.append(resumen(a, "MC-MED", "A", "original"))
  R.append(resumen(a.assign(outcome=new_y), "MC-MED", "A", "V1 reconsulta: cualquier visita con ingreso",
                   {"eventos_agregados": int((new_y & ~old_y).sum()), "eventos_perdidos": int((old_y & ~new_y).sum())}))

  sets = pd.read_csv(os.path.join(S_MC, "mcmed_sets.csv.gz"), parse_dates=["charttime"])
  sets = add_scores(sets)
  ex5 = exposure(sets, min_params=5)
  for k, base in mc.items():
      if k == "B*":
          R.append(resumen(base, "MC-MED", "B*", "original"))
      r = triage_only(base, sets, "triage"); r.update({"centro": "MC-MED", "cohorte": k}); R.append(r)
      print(f"    {r['variante']}: OR {r['OR vs estable']}")
      c5 = with_exposure(base, ex5)
      R.append(resumen(c5, "MC-MED", k, "V3 solo sets completos (5/5)", {"visitas_perdidas": int(len(base) - len(c5))}))

  # V4: ventanas fijas de 10 minutos (sin encadenar) y duracion de los sets originales
  if RUN_V4:
      print("  V4: releyendo controles de enfermeria...")
      MMAP = {"HR": "heartrate", "RR": "resprate", "SpO2": "o2sat", "SBP": "sbp", "DBP": "dbp", "Temp": "temperature"}
      parts = []
      keep = set(pd.concat([mc["A"]["stay_id"], mc["B*"]["stay_id"]]))
      for ch in pd.read_csv(os.path.join(MC, "numerics.csv"), chunksize=1_000_000, usecols=["CSN", "Source", "Measure", "Value", "Time"],
                            dtype={"Source": "category", "Measure": "category"}):
          ch = ch[(ch["Source"] == "Chart") & ch["Measure"].isin(list(MMAP)) & ch["CSN"].isin(keep)]
          parts.append(ch[["CSN", "Measure", "Value", "Time"]])
      del sets; gc.collect()
      ch = pd.concat(parts, ignore_index=True)
      ch["Time"] = fix_time(ch["Time"]); ch["Value"] = pd.to_numeric(ch["Value"], errors="coerce")
      ch = ch.dropna(subset=["Time", "Value"]).rename(columns={"CSN": "stay_id"}).sort_values(["stay_id", "Time"])
      ch["var"] = ch["Measure"].map(MMAP)
      new = (ch["stay_id"] != ch["stay_id"].shift()) | (ch["Time"] - ch["Time"].shift() > pd.Timedelta(minutes=10))
      sid = new.cumsum()
      gt = ch.groupby(sid)["Time"]
      span = (gt.max() - gt.min()).dt.total_seconds() / 60
      dur = {"sets_encadenados": int(len(span)), "duracion_>10min_%": round(100 * (span > 10).mean(), 2),
             "duracion_>30min_%": round(100 * (span > 30).mean(), 2), "duracion_p99_min": round(span.quantile(0.99), 1)}
      print(f"    sets originales: {dur}")
      # Ventanas fijas: cada set empieza en un registro y abarca como maximo 10 minutos
      t0 = ch.groupby("stay_id")["Time"].transform("min")
      ch["win"] = ((ch["Time"] - t0).dt.total_seconds() // 600).astype(int)
      st = ch.groupby(["stay_id", "win", "var"])["Value"].last().unstack("var").reset_index()
      st.columns.name = None
      st = st.merge(ch.groupby(["stay_id", "win"])["Time"].max().rename("charttime").reset_index(), on=["stay_id", "win"])
      for c_ in VITALS:
          if c_ not in st:
              st[c_] = np.nan
      trs = pd.read_csv(os.path.join(MC, "visits.csv"), low_memory=False,
                        usecols=["CSN", "Triage_Temp", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP"]) \
          .rename(columns={"CSN": "stay_id", "Triage_Temp": "temperature", "Triage_HR": "heartrate", "Triage_RR": "resprate",
                           "Triage_SpO2": "o2sat", "Triage_SBP": "sbp", "Triage_DBP": "dbp"}) \
          .merge(v[["CSN", "intime", "outtime"]].rename(columns={"CSN": "stay_id", "intime": "charttime"}), on="stay_id")
      lf = pd.concat([trs[["stay_id", "charttime"] + VITALS], st[["stay_id", "charttime"] + VITALS]], ignore_index=True)
      lf = lf.merge(v[["CSN", "intime", "outtime"]].rename(columns={"CSN": "stay_id"}), on="stay_id")
      lf = lf[(lf["charttime"] >= lf["intime"] - pd.Timedelta(minutes=30)) & (lf["charttime"] <= lf["outtime"])].sort_values(["stay_id", "charttime"])
      for c_ in VITALS:   # mismo relleno de <=60 min que el analisis principal
          lt = lf["charttime"].where(lf[c_].notna()).groupby(lf["stay_id"]).ffill()
          lf[c_] = lf[c_].fillna(lf.groupby("stay_id")[c_].ffill().where((lf["charttime"] - lt) <= pd.Timedelta(minutes=60)))
      exf = exposure(news2(lf))
      for k, base in mc.items():
          cf = with_exposure(base, exf)
          R.append(resumen(cf, "MC-MED", k, "V4 ventanas fijas de 10 min (sin encadenar)", {"visitas_perdidas": int(len(base) - len(cf)), **dur}))

df = pd.DataFrame(R)
if df.empty:
    raise SystemExit("PARTE debe ser 'mimic', 'mcmed' o 'todo'.")
cols = ["centro", "cohorte", "variante", "n_analizado", "eventos", "normalizacion_%", "tasa_normalizacion_%",
        "OR vs estable", "OR vs ultimo anormal"]
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 48)
print("\n=== Verificacion de la auditoria: modelo 2 (OR, IC95%, errores agrupados por paciente) ===")
print(df[cols].to_string(index=False))
extra = [c for c in df.columns if c not in cols]
print("\n=== Detalles ===")
print(df[["centro", "cohorte", "variante"] + extra].dropna(how="all", subset=extra).to_string(index=False))
df.to_csv(os.path.join(OUT7, f"fase7d_verificacion_{PARTE}.csv"), index=False)
print(f"\nGuardado en {OUT7}")
