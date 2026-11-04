# ============================================================================
# FALSE-STABILITY - Fase 5c: ablacion "solo ritmo" (cohorte B)
#   Se borran todos los valores vitales y puntajes NEWS2 de la secuencia.
#   Quedan solo los tiempos de cada control y las marcas de faltantes/triage,
#   mas la rama tabular. Mide cuanto predice la vigilancia por si sola.
# Requiere: fase 5b corrida en esta sesion (reutiliza tensores y predicciones).
# ============================================================================
for _need in ("Tfit", "Tval", "Tte", "zero_channels", "platt", "preds", "cal", "N_SEEDS_5B"):
    if _need not in globals():
        raise RuntimeError("Primero corre la celda de la fase 5b en esta sesion.")

VALORES = SEQ_VITALS + SEQ_SCORES
print("M3 solo ritmo (sin valores ni puntajes):")
Tf, Tv, Tt = zero_channels(Tfit, VALORES), zero_channels(Tval, VALORES), zero_channels(Tte, VALORES)
pv_all, pt_all, rows_seed = [], [], []
for s in range(N_SEEDS_5B):
    m = train_one(Tf, yfit, Tv, yval, cards, 100 + s, tag="M3 solo ritmo")
    pv, pt = predict(m, Tv), predict(m, Tt)
    pv_all.append(pv); pt_all.append(pt)
    rows_seed.append({"semilla": s, "AUPRC_toda": average_precision_score(yte, pt),
                      "AUPRC_normalizacion": average_precision_score(yte[an], pt[an])})
    del m
del Tf, Tv, Tt
torch.cuda.empty_cache()
f_cal = platt(np.mean(pv_all, 0), yval)
p_ritmo = f_cal(np.mean(pt_all, 0))

comp = {"M2_gbm": preds["M2_gbm"],
        "M3 completo (recal)": preds["M3 completo (recal)"],
        "M3 sin tiempos (recal)": preds["M3 sin tiempos (recal)"],
        "M3 solo ritmo (recal)": p_ritmo}

rows = []
for subset, msk in (("toda la prueba", np.ones(len(yte), bool)), ("normalizacion aparente", an)):
    y = yte[msk]
    P = {k: p[msk] for k, p in comp.items()}
    ci_full = boot_ci(y, P, ref="M3 completo (recal)")
    ci_val = boot_ci(y, P, ref="M3 sin tiempos (recal)")
    for k, p in P.items():
        m = metrics(y, p)
        rows.append({"subgrupo": subset, "modelo": k, "AUROC": round(m["AUROC"], 3),
                     "AUPRC": round(m["AUPRC"], 3), "AUPRC_IC95": ci_full[k]["AUPRC_ic"],
                     "O/E": round(m["O/E"], 2),
                     "dAUPRC_vs_M3_completo": ci_full[k].get("dAUPRC_vs_M3 completo (recal)", ""),
                     "dAUPRC_vs_sin_tiempos": ci_val[k].get("dAUPRC_vs_M3 sin tiempos (recal)", "")})
res_ritmo = pd.DataFrame(rows)

sd = pd.DataFrame(rows_seed)
seed_ritmo = pd.DataFrame([{"variante": "M3 solo ritmo",
                            "AUPRC_toda_media": round(sd["AUPRC_toda"].mean(), 3),
                            "AUPRC_toda_DE": round(sd["AUPRC_toda"].std(), 3),
                            "AUPRC_normalizacion_media": round(sd["AUPRC_normalizacion"].mean(), 3),
                            "AUPRC_normalizacion_DE": round(sd["AUPRC_normalizacion"].std(), 3),
                            "AUPRC_normalizacion_min": round(sd["AUPRC_normalizacion"].min(), 3),
                            "AUPRC_normalizacion_max": round(sd["AUPRC_normalizacion"].max(), 3)}])

for titulo, df, f in (("Solo ritmo: 5 semillas (AUPRC en prueba)", pd.concat([seed_sum, seed_ritmo]), "fase5c_semillas"),
                      ("Solo ritmo vs valores vs completo (ensambles recalibrados, IC95 bootstrap)", res_ritmo, "fase5c_desempeno")):
    print(f"\n=== {titulo} ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, f + ".csv"), index=False)
