# ============================================================================
# FALSE-STABILITY - Fase 5b: robustez de M3 en la cohorte B
#   1. Separar los canales de tiempo: que parte del tiempo explica la ventaja
#   2. 5 semillas por variante (ensamble + variabilidad entre semillas)
#   3. Recalibracion (Platt en validacion) y curva de decision
# Requiere: fases 3 y 5 corridas en esta sesion (GPU).
# Todo lo que imprime y los graficos son agregados y se pueden compartir.
# ============================================================================
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

for _need in ("long_seq", "train_one", "predict", "build_seq", "build_tab", "tab_encoders", "seq_stats",
              "split_val", "to_dev", "SEQ_CH", "fit_m0", "fit_gbm", "decision_curve", "discordance", "boot_ci"):
    if _need not in globals():
        raise RuntimeError("Primero corre las celdas de las fases 3 y 5 en esta sesion.")

N_SEEDS_5B = 5
VARIANTES = {
    # nombre: canales de tiempo que se ponen en cero
    "M3 completo": [],
    "M3 sin hora de salida": ["t_exit"],                 # quita estadia total y horas al traslado
    "M3 solo intervalos": ["t_exit", "t_arr"],           # deja solo el espaciamiento entre controles
    "M3 sin tiempos": ["t_exit", "t_arr", "dt_prev"],
}


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


# ----------------------------------------------------------------------------
# Datos cohorte B (misma particion que la fase 5)
# ----------------------------------------------------------------------------
nombre = "B: UCI 48h"
c = prep(load_out("cohorte_B")).reset_index(drop=True)
tr = c[c["split"] == "train"].reset_index(drop=True)
te = c[c["split"] == "test"].reset_index(drop=True)
vmask = split_val(tr)
fitd, vald = tr[~vmask].reset_index(drop=True), tr[vmask].reset_index(drop=True)
stats = seq_stats(long_seq, fitd["stay_id"])
enc = tab_encoders(fitd)
cards = [len(enc[3][k]) + 1 for k in CAT]


def tensors(d):
    X, ln = build_seq(d, long_seq, stats)
    xn, xc = build_tab(d, enc)
    return to_dev(X, ln, xn, xc)


Tfit, Tval, Tte = tensors(fitd), tensors(vald), tensors(te)
yfit, yval, yte = fitd["y"].to_numpy(), vald["y"].to_numpy(), te["y"].to_numpy()
an = te["apparent_normalization"].to_numpy()

print("Referencias (M0 NEWS2, M2 gradient boosting)...")
preds = {"M0_news2": fit_m0(tr)(te), "M2_gbm": fit_gbm(tr, FEATS["M2"])(te)}

# ----------------------------------------------------------------------------
# Entrenamiento: 5 semillas por variante
# ----------------------------------------------------------------------------
seed_rows, raw, cal, cal_val = [], {}, {}, {}
for vname, zeros in VARIANTES.items():
    print(f"\n{vname}:")
    Tf, Tv, Tt = zero_channels(Tfit, zeros), zero_channels(Tval, zeros), zero_channels(Tte, zeros)
    pv_all, pt_all = [], []
    for s in range(N_SEEDS_5B):
        m = train_one(Tf, yfit, Tv, yval, cards, 100 + s, tag=vname)
        pv, pt = predict(m, Tv), predict(m, Tt)
        pv_all.append(pv); pt_all.append(pt)
        seed_rows.append({"variante": vname, "semilla": s,
                          "AUPRC_toda": average_precision_score(yte, pt),
                          "AUPRC_normalizacion": average_precision_score(yte[an], pt[an])})
        del m
    pv_ens, pt_ens = np.mean(pv_all, 0), np.mean(pt_all, 0)
    raw[vname] = pt_ens
    f_cal = platt(pv_ens, yval)
    cal[vname], cal_val[vname] = f_cal(pt_ens), f_cal(pv_ens)
    del Tf, Tv, Tt
    torch.cuda.empty_cache()

for k, p in cal.items():
    preds[k + " (recal)"] = p

# ----------------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------------
seeds = pd.DataFrame(seed_rows)
seed_sum = seeds.groupby("variante", sort=False).agg(
    AUPRC_toda_media=("AUPRC_toda", "mean"), AUPRC_toda_DE=("AUPRC_toda", "std"),
    AUPRC_normalizacion_media=("AUPRC_normalizacion", "mean"),
    AUPRC_normalizacion_DE=("AUPRC_normalizacion", "std"),
    AUPRC_normalizacion_min=("AUPRC_normalizacion", "min"),
    AUPRC_normalizacion_max=("AUPRC_normalizacion", "max")).round(3).reset_index()


def tabla(y, P, subset):
    rows = []
    ci_m2 = boot_ci(y, P, ref="M2_gbm")
    ci_m3 = boot_ci(y, P, ref="M3 completo (recal)")
    for k, p in P.items():
        m = metrics(y, p)
        rows.append({"subgrupo": subset, "modelo": k, "eventos": int(y.sum()),
                     "AUROC": round(m["AUROC"], 3), "AUPRC": round(m["AUPRC"], 3),
                     "AUPRC_IC95": ci_m2[k]["AUPRC_ic"], "O/E": round(m["O/E"], 2),
                     "pendiente_cal": round(m["pendiente"], 2),
                     "dAUPRC_vs_M2": ci_m2[k].get("dAUPRC_vs_M2_gbm", ""),
                     "dAUPRC_vs_M3_completo": ci_m3[k].get("dAUPRC_vs_M3 completo (recal)", "")})
    return rows


res = pd.DataFrame(tabla(yte, preds, "toda la prueba") +
                   tabla(yte[an], {k: p[an] for k, p in preds.items()}, "normalizacion aparente"))

calib = pd.DataFrame([{"variante": k, "subgrupo": s,
                       "O/E_antes": round(yt.mean() / raw[k][msk].mean(), 2),
                       "O/E_despues": round(yt.mean() / cal[k][msk].mean(), 2),
                       "pendiente_antes": round(cal_slope(yt, raw[k][msk]), 2),
                       "pendiente_despues": round(cal_slope(yt, cal[k][msk]), 2)}
                      for k in VARIANTES
                      for s, msk in (("toda la prueba", np.ones(len(yte), bool)), ("normalizacion aparente", an))
                      for yt in [yte[msk]]])

dca_models = {"M0_news2": preds["M0_news2"], "M2_gbm": preds["M2_gbm"],
              "M3 (recal)": preds["M3 completo (recal)"]}
dca_full = decision_curve(te, dca_models, [0.02, 0.04, 0.06, 0.10, 0.15], nombre + " | toda la prueba")
dca_an = decision_curve(te[an], {k: p[an] for k, p in dca_models.items()}, [0.02, 0.03, 0.04, 0.06, 0.10],
                        nombre + " | normalizacion aparente")
dca = pd.concat([dca_full, dca_an])

# Discordancia con M3 recalibrado: umbral = percentil 90 de su riesgo en validacion
thr_m3 = np.quantile(cal_val["M3 completo"], 0.90)
disc = discordance(te, cal["M3 completo"], thr_m3, nombre).rename(columns={"riesgo_M2": "riesgo_M3"})
disc["riesgo_M3"] = disc["riesgo_M3"].str.replace("top 10% train", "top 10% en validacion")

# Grafico de calibracion M3 antes/despues + M2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (titulo, msk) in zip(axes, (("Toda la prueba", np.ones(len(yte), bool)),
                                    ("Normalizacion aparente", an))):
    for lab, p in (("M2_gbm", preds["M2_gbm"]), ("M3 sin recalibrar", raw["M3 completo"]),
                   ("M3 recalibrado", cal["M3 completo"])):
        q = pd.qcut(p[msk], 5 if titulo != "Toda la prueba" else 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p[msk], "y": yte[msk], "q": q}).groupby("q").mean()
        ax.plot(100 * g["p"], 100 * g["y"], marker="o", label=lab)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1)
    ax.set(title=f"{nombre}: {titulo}", xlabel="Riesgo predicho (%)", ylabel="Tasa observada (%)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fase5b_calibracion.png"), dpi=150)
plt.show()

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
for titulo, df, f in (("Variabilidad entre 5 semillas (AUPRC en prueba)", seed_sum, "fase5b_semillas"),
                      ("Desempeno de ensambles de 5 semillas, recalibrados (IC95 bootstrap)", res, "fase5b_desempeno"),
                      ("Calibracion antes y despues de Platt", calib, "fase5b_calibracion"),
                      ("Beneficio neto (curva de decision)", dca, "fase5b_dca"),
                      ("Discordancia: NEWS2 final normal, riesgo M3", disc, "fase5b_discordancia")):
    print(f"\n=== {titulo} ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, f + ".csv"), index=False)
