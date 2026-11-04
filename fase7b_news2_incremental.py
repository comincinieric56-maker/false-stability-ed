# ============================================================================
# FALSE-STABILITY - Fase 7b: valor incremental de la gravedad previa sobre el
# NEWS2 final (cohorte completa, ambos centros) + correccion del test de
# interaccion de la fase 7a.
#
# Modelos preespecificados (regresion logistica, terminos lineales):
#   N0  NEWS2 final
#   N1  NEWS2 final + NEWS2 maximo previo
#   N2  modelo clinico basal (edad, sexo, ESI, motivo, llegada) + NEWS2 final
#   N3  N2 + NEWS2 maximo previo          <- contraste clave: N3 vs N2
#
# Evaluacion principal: validacion cruzada de 5 particiones AGRUPADAS POR
# PACIENTE (ningun paciente en entrenamiento y prueba a la vez), con
# predicciones fuera de la particion para toda la cohorte.
# Sensibilidad: particion temporal (MIMIC: grupo de anos 2017+; MC-MED:
# particion cronologica oficial, excluyendo de la prueba a pacientes vistos en
# entrenamiento).
# IC95%: bootstrap AGRUPADO POR PACIENTE, 1.000 replicas, semilla fija.
# Se reportan n, eventos y casos excluidos por datos faltantes.
# CPU, ~25-35 min. Salida solo agregada, en salida_fase7.
# ============================================================================
import os, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

SEED = 20261103
N_BOOT = 1000
K_FOLDS = 5
UMBRALES = {"A": [0.02, 0.05], "B": [0.05, 0.10], "B*": [0.02, 0.05]}   # preespecificados por cohorte
NUM = {"N0": ["news2_last"], "N1": ["news2_last", "news2_prior_max"],
       "N2": ["age", "news2_last"], "N3": ["age", "news2_last", "news2_prior_max"]}
CAT = {"N0": [], "N1": [], "N2": ["sexo", "esi", "motivo", "llegada"], "N3": ["sexo", "esi", "motivo", "llegada"]}
CONTRASTES = [("N1", "N0"), ("N2", "N0"), ("N3", "N2")]


def load_any(folder, name):
    for ext, rd in ((".parquet", pd.read_parquet), (".csv.gz", lambda p: pd.read_csv(p, low_memory=False))):
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return rd(p)
    raise FileNotFoundError(f"{name} no esta en {folder}")


def prep(c, top_cc=30):
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
    # Sin sets previos anormales posibles: el maximo previo existe siempre (n_sets >= 2)
    c["news2_prior_max"] = pd.to_numeric(c["news2_prior_max"], errors="coerce")
    c["edad10"] = c["age"] / 10
    return c


def model(name):
    ct = ColumnTransformer([("num", StandardScaler(), NUM[name])] +
                           ([("cat", OneHotEncoder(handle_unknown="ignore"), CAT[name])] if CAT[name] else []))
    return make_pipeline(ct, LogisticRegression(C=1e4, max_iter=3000))


def oof_predictions(d):
    """Predicciones fuera de la particion, agrupando por paciente."""
    P = {m: np.full(len(d), np.nan) for m in NUM}
    gkf = GroupKFold(n_splits=K_FOLDS)
    for tr, te in gkf.split(d, d["y"], d["subject_id"]):
        for m in NUM:
            f = model(m).fit(d.iloc[tr][NUM[m] + CAT[m]], d["y"].iloc[tr])
            P[m][te] = f.predict_proba(d.iloc[te][NUM[m] + CAT[m]])[:, 1]
    return P


def holdout_predictions(tr, te):
    P = {}
    for m in NUM:
        f = model(m).fit(tr[NUM[m] + CAT[m]], tr["y"])
        P[m] = f.predict_proba(te[NUM[m] + CAT[m]])[:, 1]
    return P


# ----------------------------------------------------------------------------
# Metricas y bootstrap agrupado por paciente
# ----------------------------------------------------------------------------
def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    citl = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
    return m.params[1], citl.params[0]


def cluster_index_sampler(groups, rng):
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    uniq, starts, counts = np.unique(g_sorted, return_index=True, return_counts=True)

    def draw():
        s = rng.integers(0, len(uniq), len(uniq))
        lens = counts[s]
        base = np.repeat(starts[s], lens)
        offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        return order[base + offs]
    return draw


def evaluate(y, P, groups, rng, etiqueta):
    """Desempeno por modelo y diferencias pareadas, con IC por bootstrap agrupado."""
    draw = cluster_index_sampler(groups, rng)
    point = {m: (roc_auc_score(y, p), average_precision_score(y, p)) for m, p in P.items()}
    boot = {m: [] for m in P}
    for _ in range(N_BOOT):
        i = draw()
        if y[i].sum() == 0:
            continue
        for m, p in P.items():
            boot[m].append((roc_auc_score(y[i], p[i]), average_precision_score(y[i], p[i])))
    B = {m: np.array(v) for m, v in boot.items()}
    ci = lambda a: f"{np.percentile(a, 2.5):.3f}-{np.percentile(a, 97.5):.3f}"
    perf, diffs = [], []
    for m, p in P.items():
        sl, citl = calib(y, p)
        perf.append({**etiqueta, "modelo": m, "n": len(y), "eventos": int(y.sum()),
                     "AUROC": round(point[m][0], 3), "AUROC_IC95": ci(B[m][:, 0]),
                     "AUPRC": round(point[m][1], 3), "AUPRC_IC95": ci(B[m][:, 1]),
                     "Brier": round(brier_score_loss(y, p), 5), "pendiente_cal": round(sl, 2),
                     "CITL": round(citl, 3)})
    for a, b in CONTRASTES:
        da, dp = B[a][:, 0] - B[b][:, 0], B[a][:, 1] - B[b][:, 1]
        diffs.append({**etiqueta, "contraste": f"{a} - {b}",
                      "dAUROC": round(point[a][0] - point[b][0], 4), "dAUROC_IC95": ci(da),
                      "dAUPRC": round(point[a][1] - point[b][1], 4), "dAUPRC_IC95": ci(dp),
                      "p_bootstrap_dAUROC<=0": round((da <= 0).mean(), 4)})
    return perf, diffs


def reclasificacion(y, p_new, p_old, umbrales, etiqueta):
    rows = []
    for t in umbrales:
        new_hi, old_hi = p_new >= t, p_old >= t
        ev, ne = y == 1, y == 0
        nb = lambda hi: (hi & ev).mean() - (hi & ne).mean() * t / (1 - t)
        rows.append({**etiqueta, "umbral_%": 100 * t,
                     "eventos_suben": int((new_hi & ~old_hi & ev).sum()), "eventos_bajan": int((~new_hi & old_hi & ev).sum()),
                     "no_eventos_suben": int((new_hi & ~old_hi & ne).sum()), "no_eventos_bajan": int((~new_hi & old_hi & ne).sum()),
                     "beneficio_neto_N2_x1000": round(1000 * nb(old_hi), 2), "beneficio_neto_N3_x1000": round(1000 * nb(new_hi), 2),
                     "tratar_todos_x1000": round(1000 * (ev.mean() - ne.mean() * t / (1 - t)), 2)})
    return rows


# ----------------------------------------------------------------------------
# Correccion del test de interaccion de la fase 7a (estratos "<=4", "5-6", ">=7")
# ----------------------------------------------------------------------------
def interaction_fixed(c, sitio, coh):
    s = c[c["grupo"].isin(["normalizacion", "estable"])].copy()
    s["nsc"] = pd.cut(s["n_sets"], [1, 4, 6, np.inf], labels=["<=4", "5-6", ">=7"]).astype(str)
    cols = ["edad10", "sexo", "esi", "motivo", "llegada", "nsc", "y", "subject_id"]
    d = s.dropna(subset=cols)
    f = ("y ~ C(grupo, Treatment('estable')) * C(nsc, Treatment('<=4')) + edad10 + C(sexo) + C(esi) + C(motivo) + C(llegada)")
    try:
        m = smf.glm(f, data=d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d["subject_id"].values})
        inter = [k for k in m.params.index if ":" in k]
        R = np.zeros((len(inter), len(m.params)))
        for i, k in enumerate(inter):
            R[i, list(m.params.index).index(k)] = 1
        p = float(np.asarray(m.wald_test(R, scalar=True).pvalue))
        txt = "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")
    except Exception as e:
        txt = f"no estimable ({type(e).__name__})"
    return {"centro": sitio, "cohorte": coh, "estratos": "<=4 / 5-6 / >=7", "p_interaccion": txt,
            "eventos_expuestos_por_estrato": "; ".join(f"{k}: {int(d.loc[(d['nsc'] == k) & (d['grupo'] == 'normalizacion'), 'y'].sum())}"
                                                  for k in ["<=4", "5-6", ">=7"])}


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

R = {k: [] for k in ("desempeno", "diferencias", "reclasificacion", "interaccion", "exclusiones")}
for sitio, cohs in (("MIMIC-IV-ED", mi), ("MC-MED", mc)):
    for coh, c0 in cohs.items():
        print(f"\n### {sitio} cohorte {coh}")
        c = prep(c0)
        R["interaccion"].append(interaction_fixed(c, sitio, coh))
        cols = ["y", "subject_id", "age", "sexo", "esi", "motivo", "llegada", "news2_last", "news2_prior_max"]
        d = c.dropna(subset=cols).reset_index(drop=True)
        R["exclusiones"].append({"centro": sitio, "cohorte": coh, "n_cohorte": len(c), "n_analizado": len(d),
                                 "excluidos_por_faltantes": len(c) - len(d), "eventos": int(d["y"].sum())})
        y, g = d["y"].to_numpy(), d["subject_id"].to_numpy()

        # 1) Principal: validacion cruzada agrupada por paciente, toda la cohorte
        print("  validacion cruzada agrupada...")
        P = oof_predictions(d)
        et = {"centro": sitio, "cohorte": coh, "evaluacion": "VC 5 particiones por paciente", "subgrupo": "toda la cohorte"}
        print("  bootstrap (toda la cohorte)...")
        p_, d_ = evaluate(y, P, g, rng, et)
        R["desempeno"] += p_; R["diferencias"] += d_
        R["reclasificacion"] += reclasificacion(y, P["N3"], P["N2"], UMBRALES[coh], et)

        # 2) Secundario: subgrupo con normalizacion aparente (mismas predicciones)
        an = d["an"].to_numpy()
        et2 = {**et, "subgrupo": "normalizacion aparente"}
        p_, d_ = evaluate(y[an], {m: p[an] for m, p in P.items()}, g[an], rng, et2)
        R["desempeno"] += p_; R["diferencias"] += d_

        # 3) Sensibilidad: particion temporal sin pacientes compartidos
        if sitio == "MIMIC-IV-ED":
            te_mask = d["subject_id"].isin(mi_test_subj)
            tr_mask = ~te_mask
        else:
            sp = d["stay_id"].map(split_mc)
            tr_mask = sp.isin(["train", "val"])
            tr_subj = set(d.loc[tr_mask, "subject_id"])
            te_mask = sp.eq("test") & ~d["subject_id"].isin(tr_subj)
        tr, te = d[tr_mask], d[te_mask]
        Ph = holdout_predictions(tr, te)
        et3 = {"centro": sitio, "cohorte": coh, "evaluacion": "particion temporal (sin pacientes compartidos)", "subgrupo": "toda la cohorte"}
        print(f"  temporal: entrenamiento {len(tr):,} / prueba {len(te):,} ({int(te['y'].sum())} eventos)")
        p_, d_ = evaluate(te["y"].to_numpy(), Ph, te["subject_id"].to_numpy(), rng, et3)
        R["desempeno"] += p_; R["diferencias"] += d_

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)
titulos = {"exclusiones": "Denominadores y exclusiones por datos faltantes",
           "desempeno": "Desempeno de N0-N3 (IC95% bootstrap agrupado por paciente, 1.000 replicas)",
           "diferencias": "Diferencias pareadas (clave: N3 - N2)",
           "reclasificacion": "Reclasificacion y beneficio neto, N3 frente a N2 (umbrales preespecificados)",
           "interaccion": "Correccion del test de interaccion grupo x numero de controles (fase 7a)"}
for k, t in titulos.items():
    df = pd.DataFrame(R[k])
    print(f"\n=== {t} ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT7, f"fase7b_{k}.csv"), index=False)
print(f"\nGuardado en {OUT7}")
