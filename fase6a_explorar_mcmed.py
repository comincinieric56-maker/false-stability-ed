# ============================================================================
# FALSE-STABILITY - Fase 6a: exploracion de MC-MED (solo agregados)
# Imprime categorias, conteos y percentiles para adaptar las cohortes.
# No imprime filas de pacientes: la salida se puede compartir.
# Requiere Drive montado. ~5-10 minutos (numerics.csv se lee por partes).
# ============================================================================
import os
import numpy as np
import pandas as pd

MC = globals().get("dest", "/content/drive/MyDrive/ARTICULOS MIOS/False Stability/mcmed")
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def vc(s, n=25):
    t = s.fillna("<vacio>").astype(str).value_counts()
    return t.head(n).to_string() + (f"\n  ... {len(t) - n} categorias mas" if len(t) > n else "")


def pct(s):
    s = pd.to_numeric(s, errors="coerce")
    q = s.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).round(2)
    return f"n={s.notna().sum():,} faltantes={s.isna().mean():.1%} | " + \
           " ".join(f"p{int(k * 100)}={v}" for k, v in q.items())


def parse_ok(s):
    t = pd.to_datetime(s, errors="coerce")
    return f"parse ok {t.notna().mean():.1%}, rango anos {t.dt.year.min()}-{t.dt.year.max()}"


# ---------------------------------------------------------------------------
print("=" * 30, "visits.csv", "=" * 30)
v = pd.read_csv(os.path.join(MC, "visits.csv"), low_memory=False)
print(f"visitas {len(v):,} | pacientes {v['MRN'].nunique():,} | CSN unicos {v['CSN'].nunique():,}")
for col in ["ED_dispo", "Dispo_class_next_visit", "DC_dispo", "Means_of_arrival", "Triage_acuity",
            "Gender", "Admit_service", "Payor_class"]:
    print(f"\n--- {col} ---\n{vc(v[col])}")
print("\n--- Numericas ---")
for col in ["Age", "Visits", "Visit_no", "ED_LOS", "Hosp_LOS", "Hours_to_next_visit",
            "Triage_Temp", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP"]:
    print(f"{col:22s} {pct(v[col])}")
print("\n--- Fechas ---")
for col in ["Arrival_time", "Roomed_time", "Dispo_time", "Admit_time", "Departure_time"]:
    print(f"{col:16s} {parse_ok(v[col])}")
a, d = pd.to_datetime(v["Arrival_time"], errors="coerce"), pd.to_datetime(v["Departure_time"], errors="coerce")
print("ED_LOS vs (Departure-Arrival) en horas, mediana de la diferencia:",
      round(((d - a).dt.total_seconds() / 3600 - pd.to_numeric(v["ED_LOS"], errors="coerce")).median(), 3))
nv = pd.to_numeric(v["Hours_to_next_visit"], errors="coerce")
print("\nReconsulta <=72 h:", f"{(nv <= 72).mean():.2%}",
      "| destino de esas reconsultas:\n" + vc(v.loc[nv <= 72, "Dispo_class_next_visit"], 15))
print("\nCruce ED_dispo x DC_dispo (top):")
print(pd.crosstab(v["ED_dispo"].fillna("<vacio>"), v["DC_dispo"].fillna("<vacio>")).loc[:, lambda t: t.sum().sort_values(ascending=False).index[:10]])

# ---------------------------------------------------------------------------
print("\n" + "=" * 30, "splits", "=" * 30)
for s in ("train", "val", "test"):
    ids = pd.read_csv(os.path.join(MC, f"split_chrono_{s}.csv"), header=None)[0]
    print(f"{s}: {len(ids):,} CSN | en visits: {ids.isin(v['CSN']).mean():.1%}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 30, "numerics.csv (por partes)", "=" * 30)
counts, n_rows, csns = {}, 0, set()
samples = {}
first_chunk = None
for i, ch in enumerate(pd.read_csv(os.path.join(MC, "numerics.csv"), chunksize=5_000_000,
                                   usecols=["CSN", "Source", "Measure", "Value", "Time"])):
    n_rows += len(ch)
    csns.update(ch["CSN"].unique())
    g = ch.groupby([ch["Source"].fillna("<vacio>"), ch["Measure"].fillna("<vacio>")]).size()
    for k, val in g.items():
        counts[k] = counts.get(k, 0) + val
    if i < 3:  # muestra para percentiles y frecuencia
        for (src, ms), sub in ch.groupby(["Source", "Measure"]):
            samples.setdefault((src, ms), []).append(sub)
    if i == 0:
        first_chunk = ch
    print(f"  parte {i + 1}: {n_rows:,} filas acumuladas", flush=True)

print(f"\nfilas {n_rows:,} | visitas con numerics {len(csns):,} ({len(csns) / v['CSN'].nunique():.1%} de visits)")
print("Time:", parse_ok(first_chunk["Time"]))
cnt = pd.Series(counts).sort_values(ascending=False)
print("\n--- Filas por Source x Measure ---")
print(cnt.to_string())

print("\n--- Por Source x Measure (muestra de las 3 primeras partes) ---")
rows = []
for (src, ms), parts in samples.items():
    s = pd.concat(parts)
    s = s.assign(t=pd.to_datetime(s["Time"], errors="coerce"), val=pd.to_numeric(s["Value"], errors="coerce"))
    s = s.sort_values(["CSN", "t"])
    gap = s.groupby("CSN")["t"].diff().dt.total_seconds() / 60
    per = s.groupby("CSN").size()
    q = s["val"].quantile([.01, .5, .99])
    rows.append({"Source": src, "Measure": ms, "filas_muestra": len(s), "visitas": s["CSN"].nunique(),
                 "mediciones/visita (mediana)": per.median(),
                 "minutos entre mediciones (mediana)": round(gap.median(), 1),
                 "p01": round(q.loc[.01], 2), "p50": round(q.loc[.5], 2), "p99": round(q.loc[.99], 2),
                 "no numerico": f"{s['val'].isna().mean():.1%}"})
print(pd.DataFrame(rows).sort_values(["Source", "filas_muestra"], ascending=[True, False]).to_string(index=False))

# ---------------------------------------------------------------------------
print("\n" + "=" * 30, "orders.csv", "=" * 30)
o = pd.read_csv(os.path.join(MC, "orders.csv"), low_memory=False)
print(f"filas {o.shape[0]:,} | visitas {o['CSN'].nunique():,}")
print("\n--- Order_type ---\n" + vc(o["Order_type"], 30))
print("\nFirst_admin_time:", parse_ok(o["First_admin_time"]), f"| no vacio {o['First_admin_time'].notna().mean():.1%}")
med_type = o["Order_type"].astype(str).str.contains("med", case=False)
print("\n--- Procedure_name mas frecuentes en ordenes de tipo medicamento ---")
print(vc(o.loc[med_type, "Procedure_name"], 60))
