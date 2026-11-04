# ============================================================================
# FALSE-STABILITY - Fase 2b: analisis ajustado y de sensibilidad
# Usa los archivos de 'salida' generados en la fase 2. Todo lo que imprime es
# agregado (n, tasas, OR) y se puede compartir.
# ============================================================================
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
FOLDER = globals().get("FOLDER", "/content/drive/MyDrive/ARTICULOS MIOS/False Stability")
OUT = globals().get("OUT", FOLDER + "/salida")


def load(n):
    p = os.path.join(OUT, n + ".parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.read_csv(os.path.join(OUT, n + ".csv.gz"))


def prep(c, top_cc=30):
    c = c.copy()
    c["grupo"] = np.select(
        [c["apparent_normalization"], c["stable_throughout"]],
        ["normalizacion", "estable"], default="ultimo_anormal")
    c["y"] = c["outcome"].astype(int)
    c["sexo"] = c["gender"].fillna("NA").astype(str)
    c["esi"] = c["acuity"].fillna(-1).astype(int).astype(str).replace("-1", "NA")
    c["llegada"] = c["arrival_transport"].fillna("NA").astype(str)
    cc = c["chiefcomplaint"].fillna("NA").astype(str).str.lower().str.strip()
    top = cc.value_counts().index[:top_cc]
    c["motivo"] = np.where(cc.isin(top), cc, "otro")
    c["edad10"] = c["age"] / 10
    c["log_sets"] = np.log(c["n_sets"])
    c["los_h"] = c["ed_los_h"].clip(0, 72)
    c["h_ult"] = c["hours_last_set_to_exit"].clip(0, 48)
    return c


MODELOS = {
    "M1 edad+sexo+ESI": "y ~ C(grupo, Treatment('estable')) + edad10 + C(sexo) + C(esi)",
    "M2 + motivo+llegada": "y ~ C(grupo, Treatment('estable')) + edad10 + C(sexo) + C(esi)"
                           " + C(motivo) + C(llegada)",
    "M3 + proceso": "y ~ C(grupo, Treatment('estable')) + edad10 + C(sexo) + C(esi)"
                    " + C(motivo) + C(llegada) + log_sets + los_h + h_ult",
}


def ajustado(c, nombre):
    rows = []
    for mname, f in MODELOS.items():
        cols = ["y", "grupo", "edad10", "sexo", "esi", "motivo", "llegada",
                "log_sets", "los_h", "h_ult", "subject_id"]
        d = c[cols].dropna()  # los grupos del SE deben calzar fila a fila
        m = smf.glm(f, data=d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d["subject_id"].values})
        for g in ("normalizacion", "ultimo_anormal"):
            k = f"C(grupo, Treatment('estable'))[T.{g}]"
            if k not in m.params:
                continue
            b, se = m.params[k], m.bse[k]
            rows.append({"cohorte": nombre, "modelo": mname, "grupo vs estable": g,
                         "OR": round(np.exp(b), 2),
                         "IC95": f"{np.exp(b - 1.96 * se):.2f}-{np.exp(b + 1.96 * se):.2f}"})
    return pd.DataFrame(rows)


def tabla1(c, nombre):
    g = c.groupby("grupo")
    t = pd.DataFrame({
        "n": g.size(),
        "edad_mediana": g["age"].median(),
        "pct_mujer": g["sexo"].apply(lambda s: round(100 * (s == "F").mean(), 1)),
        "pct_ESI_1_2": g["esi"].apply(lambda s: round(100 * s.isin(["1", "2"]).mean(), 1)),
        "sets_mediana": g["n_sets"].median(),
        "news2_max_mediana": g["news2_max"].median(),
        "estadia_h_mediana": g["ed_los_h"].median().round(1),
        "h_ultimo_set_mediana": g["hours_last_set_to_exit"].median().round(2),
        "tasa_pct": (100 * g["y"].mean()).round(2),
    })
    t.insert(0, "cohorte", nombre)
    return t.reset_index()


def rate(sub):
    n, e = len(sub), int(sub["y"].sum())
    return n, e, round(100 * e / max(n, 1), 2)


def sensibilidad(c, nombre):
    """Definiciones alternativas de normalizacion aparente, comparadas con estables."""
    last_norm = (c["news2_last"] <= 1) & (c["maxcomp_last"] < 2)
    pm = c["news2_prior_max"].fillna(-1)
    defs = {
        "Principal (previo >=3 o param=3; ultimo <=1)": (c["apparent_normalization"], c["stable_throughout"]),
        "Previo NEWS2 >=5; ultimo <=1": (last_norm & (pm >= 5), last_norm & (pm < 3)),
        "Previo NEWS2 >=3; ultimo = 0": ((c["news2_last"] == 0) & (pm >= 3), (c["news2_last"] == 0) & (pm < 3)),
    }
    rows = []
    for d, (exp, ref) in defs.items():
        n1, e1, r1 = rate(c[exp]); n0, e0, r0 = rate(c[ref])
        rr = (e1 / n1) / (e0 / n0) if n1 and n0 and e0 else np.nan
        se = np.sqrt(1 / e1 - 1 / n1 + 1 / e0 - 1 / n0) if e1 and e0 else np.nan
        rows.append({"cohorte": nombre, "definicion": d, "n_expuestos": n1, "eventos": e1,
                     "tasa_exp_pct": r1, "tasa_estables_pct": r0, "RR": round(rr, 2),
                     "IC95": f"{rr * np.exp(-1.96 * se):.2f}-{rr * np.exp(1.96 * se):.2f}"})
    return pd.DataFrame(rows)


def por_ausencia_controles(c, nombre):
    """Eje 'ausencia de controles': tasa segun horas entre ultimo set y salida."""
    c = c.copy()
    c["h_cat"] = pd.cut(c["hours_last_set_to_exit"], [-0.01, 1, 3, 6, np.inf],
                        labels=["<=1 h", "1-3 h", "3-6 h", ">6 h"])
    t = c[c["grupo"] != "ultimo_anormal"].groupby(["h_cat", "grupo"], observed=True)["y"] \
        .agg(n="size", eventos="sum", tasa_pct=lambda s: round(100 * s.mean(), 2)).reset_index()
    t.insert(0, "cohorte", nombre)
    return t


pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 20)
res = {}
for n, nombre in (("cohorte_A", "A: reconsulta 72h"), ("cohorte_B", "B: UCI 48h")):
    c = prep(load(n))
    res[nombre] = dict(t1=tabla1(c, nombre), aj=ajustado(c, nombre),
                       sens=sensibilidad(c, nombre), aus=por_ausencia_controles(c, nombre))

for titulo, key in (("Tabla 1 por grupo", "t1"), ("OR ajustados (referencia: estables, SE por paciente)", "aj"),
                    ("Sensibilidad de la definicion", "sens"), ("Ausencia de controles", "aus")):
    print(f"\n=== {titulo} ===")
    df = pd.concat([r[key] for r in res.values()])
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, f"fase2b_{key}.csv"), index=False)
