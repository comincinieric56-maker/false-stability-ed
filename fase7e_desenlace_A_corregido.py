# ============================================================================
# FALSE-STABILITY - Fase 7e: adopta el desenlace A corregido como PRINCIPAL
#   Desenlace A = CUALQUIER reconsulta a urgencias <=72 h desde la salida que
#   termine en ingreso (antes: solo la primera reconsulta).
#   1. Respalda los archivos de cohorte A originales (*_primera_reconsulta).
#   2. Reescribe cohorte_A y mcmed_cohorte_A con el desenlace corregido
#      (la columna outcome_primera_reconsulta conserva el anterior).
#   Despues, volver a ejecutar en este orden: 7a, 7b, 7c (parte 1), 7c (parte 2).
#   Las cohortes B y B* no cambian. CPU, ~3 min. Ejecutar UNA sola vez.
# ============================================================================
import os, re, zipfile, shutil
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

FOLDER = globals().get("FOLDER", "/content/drive/MyDrive/ARTICULOS MIOS/False Stability")
S_MI, S_MC, MC = (os.path.join(FOLDER, x) for x in ("salida", "salida_mcmed", "mcmed"))
ED_ZIP = os.path.join(FOLDER, "mimic-iv-ed-2.2.zip")
H = pd.Timedelta(hours=72)


def path_of(folder, name):
    for ext in (".parquet", ".csv.gz"):
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def read(p):
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, low_memory=False)


def write(df, p):
    df.to_parquet(p, index=False) if p.endswith(".parquet") else df.to_csv(p, index=False)


def backup(p):
    b = p.replace(".parquet", "_primera_reconsulta.parquet").replace(".csv.gz", "_primera_reconsulta.csv.gz")
    if os.path.exists(b):
        raise RuntimeError(f"Ya existe {os.path.basename(b)}: esta fase ya se ejecuto. No la repitas.")
    shutil.copy2(p, b)
    return b


def any_return(idx, visits, subj, t_in, t_out, admitted):
    """idx: stay_id, subject_id, outtime. visits: subject_id, n_in, admitted (bool)."""
    m = idx.merge(visits, on="subject_id")
    m = m[(m["n_in"] > m["outtime"]) & (m["n_in"] <= m["outtime"] + H)]
    return m.groupby("stay_id")["adm"].any()


# ---------------- MIMIC-IV-ED ----------------
with zipfile.ZipFile(ED_ZIP) as z:
    n = [x for x in z.namelist() if os.path.basename(x).startswith("edstays.csv")][0]
    with z.open(n) as f:
        ed = pd.read_csv(f, compression="gzip" if n.endswith(".gz") else None,
                         usecols=["subject_id", "stay_id", "intime", "outtime", "hadm_id"], parse_dates=["intime", "outtime"])
p = path_of(S_MI, "cohorte_A")
a = read(p)
idx = a[["stay_id", "subject_id"]].merge(ed[["stay_id", "outtime"]], on="stay_id")
vis = ed[["subject_id", "intime", "hadm_id"]].rename(columns={"intime": "n_in"}).assign(adm=lambda d: d["hadm_id"].notna())
y = a["stay_id"].map(any_return(idx, vis[["subject_id", "n_in", "adm"]], None, None, None, None)).fillna(False).astype(bool)
print(f"MIMIC A: eventos {int(a['outcome'].astype(bool).sum()):,} -> {int(y.sum()):,}")
b = backup(p)
a["outcome_primera_reconsulta"] = a["outcome"]
a["outcome"] = y
write(a, p)
print(f"  respaldo: {os.path.basename(b)} | corregido: {os.path.basename(p)}")

# ---------------- MC-MED ----------------
YEAR_RE = re.compile(r"\b(2[1-4]\d\d)\b")


def fix_time(s):
    s = s.astype("string")
    f = s.str.replace(YEAR_RE, lambda m_: str(int(m_.group(1)) - 400), regex=True)
    t = pd.to_datetime(f, format="ISO8601", errors="coerce")
    if t.notna().mean() < 0.99 * s.notna().mean():
        t = t.fillna(pd.to_datetime(f, format="mixed", errors="coerce"))
    return t


v = pd.read_csv(os.path.join(MC, "visits.csv"), low_memory=False, usecols=["MRN", "CSN", "Arrival_time", "Departure_time", "ED_dispo"])
v["intime"], v["outtime"] = fix_time(v["Arrival_time"]), fix_time(v["Departure_time"])
p = path_of(S_MC, "mcmed_cohorte_A")
a = read(p)
idx = a[["stay_id", "subject_id"]].merge(v[["CSN", "outtime"]].rename(columns={"CSN": "stay_id"}), on="stay_id")
vis = v.rename(columns={"MRN": "subject_id", "intime": "n_in"})[["subject_id", "n_in"]].assign(
    adm=v["ED_dispo"].isin(["Inpatient", "Observation", "ICU"]).values)
y = a["stay_id"].map(any_return(idx, vis, None, None, None, None)).fillna(False).astype(bool)
print(f"MC-MED A: eventos {int(a['outcome'].astype(bool).sum()):,} -> {int(y.sum()):,}")
b = backup(p)
a["outcome_primera_reconsulta"] = a["outcome"]
a["outcome"] = y
write(a, p)
print(f"  respaldo: {os.path.basename(b)} | corregido: {os.path.basename(p)}")
print("\nListo. Ahora ejecuta 7a, 7b, 7c (parte 1) y 7c (parte 2), en ese orden, y pega las salidas.")
