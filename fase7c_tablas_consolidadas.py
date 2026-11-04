# ============================================================================
# FALSE-STABILITY - Fase 7c (parte 2): figuras y tablas consolidadas, articulo 1
#   Ejecutar DESPUES de fase7c_sensibilidad_calibracion.py. Solo lee resultados
#   agregados ya guardados en Drive. CPU, ~1 minuto.
#
#   Carpeta nueva: articulo1_figuras_tablas_v2 (la version anterior no se toca)
#   Texto principal (7 piezas):
#     Figura 1  Diagrama de flujo
#     Figura 2  Asociacion ajustada (modelo 2 principal; M1/M3 sensibilidad;
#               contraste directo frente a ultimo set anormal)
#     Figura 3  Sensibilidad (frecuencia de medicion, triage, definiciones,
#               oxigeno, reconstruccion de MC-MED)
#     Figura 4  Calibracion en normalizacion aparente: N2 frente a N3
#               (VC agrupada y particion temporal)
#     Tabla 1   Caracteristicas (visitas y pacientes unicos)
#     Tabla 2   Tasas y OR ajustados
#     Tabla 3   Valor incremental de la gravedad previa (N0-N4)
#   Suplemento S1-S8 y trazabilidad.csv (pieza -> archivo -> script).
# ============================================================================
import os, re, subprocess, sys, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.patches import FancyBboxPatch

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass
try:
    import docx
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "python-docx"], check=True)
    import docx
from docx.shared import Pt
from docx.enum.section import WD_ORIENT

FOLDER = globals().get("FOLDER", "/content/drive/MyDrive/ARTICULOS MIOS/False Stability")
S_MI, S_MC, S7 = (os.path.join(FOLDER, x) for x in ("salida", "salida_mcmed", "salida_fase7"))
DEST = os.path.join(FOLDER, "articulo1_figuras_tablas_v2")
os.makedirs(DEST, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 110})
AZUL, GRIS, GRIS_CL, NEGRO = "#2563eb", "#6b7280", "#d1d5db", "#111827"
COHORTES = [("MIMIC-IV-ED", "A", "Cohorte A · reconsulta con ingreso ≤72 h"),
            ("MC-MED", "A", "Cohorte A · reconsulta con ingreso ≤72 h"),
            ("MIMIC-IV-ED", "B", "Cohorte B · UCI o muerte en sala ≤48 h"),
            ("MC-MED", "B*", "Cohorte B* · muerte intrahospitalaria u hospicio")]
SCRIPT = {"salida": "false_stability_cohort.py", "salida_mcmed": "fase6b_mcmed_fenomeno.py"}
traza, hechos = [], []


def leer(carpeta, archivo):
    p = os.path.join(carpeta, archivo)
    if not os.path.exists(p):
        print(f"  (falta {archivo}: se omite lo que depende de el)")
        return None
    return pd.read_csv(p)


def fuente(pieza, carpeta, archivo):
    base = os.path.basename(carpeta)
    scr = SCRIPT.get(base) or re.sub(r"^(fase7[abc]).*", r"\1", archivo)
    scr = {"fase7a": "fase7a_epidemiologia.py", "fase7b": "fase7b_news2_incremental.py",
           "fase7c": "fase7c_sensibilidad_calibracion.py"}.get(scr, scr)
    traza.append({"pieza": pieza, "archivo_fuente": f"{base}/{archivo}", "script": scr})


def es(x):
    """Formato espanol: coma decimal y guion largo en intervalos."""
    s = str(x)
    if s in ("nan", "None"):
        return ""
    s = re.sub(r"(?<=\d)-(?=-?\d)", "–", s)
    s = re.sub(r"(\d)\.(\d)", r"\1,\2", s)
    return s.replace(" a ", " a ")


def num(s):
    m = re.match(r"\s*(-?\d+\.?\d*)", str(s))
    return float(m.group(1)) if m else np.nan


def ic(s):
    m = re.findall(r"(-?\d+\.?\d*)\s*(?:-|a|–)\s*(-?\d+\.?\d*)", str(s))
    return (float(m[-1][0]), float(m[-1][1])) if m else (np.nan, np.nan)


def coh_key(s):
    s = str(s).strip().upper()
    return "A" if s.startswith("A") else ("B*" if s.startswith("B*") else "B")


def sel(df, sitio, coh, **kw):
    d = df[(df["centro"] == sitio) & (df["cohorte"].map(coh_key) == coh)]
    for k, v in kw.items():
        d = d[d[k] == v]
    return d


def guardar_fig(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DEST, f"{nombre}.{ext}"), dpi=300)
    plt.close(fig)
    hechos.append(nombre)
    print(f"  {nombre} lista")


def est_ic(x, intervalo, d=3):
    """Estimacion (IC) con decimales fijos y separador ' a ' en todas las tablas."""
    lo, hi = ic(intervalo)
    if pd.isna(x) or np.isnan(lo):
        return "no estimable"
    return es(f"{float(x):.{d}f} ({lo:.{d}f} a {hi:.{d}f})")


def miles(n):
    return f"{int(n):,}".replace(",", ".")


# --- Lectura -----------------------------------------------------------------
flu_mi, flu_mc = leer(S_MI, "flujo_seleccion.csv"), leer(S_MC, "fase6b_flujo.csv")
con = leer(S7, "fase7a_contrastes.csv")
fre = leer(S7, "fase7a_frecuencia.csv")
dfs = leer(S7, "fase7a_definiciones.csv")
rec = leer(S7, "fase7a_reconstruccion_mcmed.csv")
fen = leer(S7, "fase7a_fenotipos.csv")
fal = leer(S7, "fase7a_faltantes.csv")
b_des, b_dif = leer(S7, "fase7b_desempeno.csv"), leer(S7, "fase7b_diferencias.csv")
b_rec, b_int = leer(S7, "fase7b_reclasificacion.csv"), leer(S7, "fase7b_interaccion.csv")
b_exc = leer(S7, "fase7b_exclusiones.csv")
c_t1, c_mu = leer(S7, "fase7c_tabla1.csv"), leer(S7, "fase7c_muestra.csv")
c_des, c_dif, c_cur = leer(S7, "fase7c_desempeno.csv"), leer(S7, "fase7c_diferencias.csv"), leer(S7, "fase7c_curvas.csv")
VC, TEMP = "VC 5 particiones por paciente", "particion temporal (sin pacientes compartidos)"

# ============================================================================
# FIGURA 1: flujo
# ============================================================================
if flu_mi is not None and flu_mc is not None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (titulo, flu) in zip(axes, (("MIMIC-IV-ED (Boston, 2011-2019)", flu_mi), ("MC-MED (Stanford, 2020-2022)", flu_mc))):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.set_title(titulo, fontsize=10, fontweight="bold", color=NEGRO)
        for k, x0 in (("A", 0.2), ("B", 5.2)):
            pasos = flu[flu["cohorte"].map(lambda s: "A" if coh_key(s) == "A" else "B") == k].reset_index(drop=True)
            n = len(pasos)
            ys = np.linspace(8.8, 1.2, max(n, 2))[:n]
            for i, r in pasos.iterrows():
                final = i == n - 1
                ax.add_patch(FancyBboxPatch((x0, ys[i] - 0.55), 4.6, 1.1, boxstyle="round,pad=0.05,rounding_size=0.15",
                                            fc=AZUL if final else "white", ec=AZUL if final else GRIS, lw=1))
                paso = str(r["paso"]).replace(">=", "≥")
                ax.text(x0 + 2.3, ys[i] + 0.15, textwrap.fill(paso, 30), ha="center", va="center",
                        fontsize=7, color="white" if final else NEGRO, linespacing=1.0)
                ax.text(x0 + 2.3, ys[i] - 0.33, f"n = {miles(r['n'])}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white" if final else NEGRO)
                if i < n - 1:
                    ax.annotate("", xy=(x0 + 2.3, ys[i + 1] + 0.6), xytext=(x0 + 2.3, ys[i] - 0.6),
                                arrowprops=dict(arrowstyle="->", color=GRIS, lw=1))
            et = "Cohorte A: alta" if k == "A" else ("Cohorte B: sala" if "MIMIC" in titulo else "Cohorte B*: sala/observación")
            ax.text(x0 + 2.3, 9.75, et, ha="center", fontsize=8.5, color=GRIS, fontweight="bold")
    guardar_fig(fig, "Figura1_flujo")
    fuente("Figura 1", S_MI, "flujo_seleccion.csv"); fuente("Figura 1", S_MC, "fase6b_flujo.csv")

# ============================================================================
# FIGURA 2: asociacion ajustada
# ============================================================================
if con is not None:
    filas_fig = [("normalizacion vs estable", "M2 principal", "vs estables · modelo 2 (principal)", AZUL, "o", True),
                 ("normalizacion vs estable", "M1 sensibilidad", "vs estables · modelo 1", AZUL, "o", False),
                 ("normalizacion vs estable", "M3 sensibilidad (proceso)", "vs estables · modelo 3 (+ proceso)", AZUL, "o", False),
                 ("normalizacion vs ultimo set anormal", "M2 principal", "vs último set anormal · modelo 2", GRIS, "s", True)]
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    y, yt, yl = 0, [], []
    for sitio, coh, et in COHORTES[::-1]:
        for cont, mod, lab, col, mk, lleno in filas_fig[::-1]:
            r = sel(con, sitio, coh, contraste=cont, modelo=mod)
            if r.empty:
                continue
            o, (lo, hi) = num(r["OR (IC95%)"].iloc[0]), ic(r["OR (IC95%)"].iloc[0])
            ax.plot([lo, hi], [y, y], color=col, lw=1.4 if lleno else 0.9)
            ax.plot(o, y, mk, color=col, ms=5.5 if lleno else 4.5, mfc=col if lleno else "white")
            if lleno:
                ax.text(4.3, y, f"{es(f'{o:.2f}')} ({es(f'{lo:.2f}')}–{es(f'{hi:.2f}')})", va="center", fontsize=7.4, color=col)
            yt.append(y); yl.append(lab); y += 1
        ax.text(0.33, y - 0.2, f"{et} · {sitio}", fontsize=8.3, fontweight="bold", color=NEGRO)
        y += 1.1
    ax.axvline(1, color=NEGRO, lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.33, 4.2); ax.set_xticks([0.4, 0.5, 0.75, 1, 1.5, 2, 3])
    ax.set_xticklabels(["0,4", "0,5", "0,75", "1", "1,5", "2", "3"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.4); ax.set_ylim(-0.8, y)
    ax.set_xlabel("OR ajustado de la normalización aparente (escala log; errores agrupados por paciente)")
    ax.text(4.3, y - 0.2, "OR (IC95%)", fontsize=7.4, fontweight="bold", color=NEGRO)
    ax.set_title("Más riesgo que los estables en ambos centros; al alta, riesgo no distinguible\n"
                 "del de salir con signos anormales; en sala, menor", fontsize=10, loc="left", color=NEGRO)
    guardar_fig(fig, "Figura2_asociacion_ajustada")
    fuente("Figura 2", S7, "fase7a_contrastes.csv")

# ============================================================================
# FIGURA 3: sensibilidad (todas OR M2 frente a estables)
# ============================================================================
if con is not None and fre is not None and dfs is not None:
    def filas_sens(sitio, coh):
        out = []
        r = sel(con, sitio, coh, contraste="normalizacion vs estable", modelo="M2 principal")
        if not r.empty:
            out.append(("Análisis principal", r["OR (IC95%)"].iloc[0], NEGRO))
        for pat, lab in (("M2 \\+ categoria", "+ ajuste por n.º de controles"), ("solo triage", "Definición solo con el triage")):
            r = sel(fre, sitio, coh); r = r[r["analisis"].str.contains(pat, regex=True)]
            if not r.empty:
                out.append((lab, r["OR M2 (IC95%)"].iloc[0], GRIS))
        for pat, lab in (("previo NEWS2 >=5", "Previo NEWS2 ≥5"), ("ultimo = 0", "Último NEWS2 = 0"),
                         ("sin oxigeno", "Excluye oxígeno suplementario")):
            r = sel(dfs, sitio, coh); r = r[r["definicion"].str.contains(pat, regex=False)]
            if not r.empty:
                out.append((lab, r["OR M2 (IC95%)"].iloc[0], AZUL if ">=5" in pat else GRIS))
        if rec is not None and sitio == "MC-MED":
            for _, r in sel(rec, sitio, coh).iterrows():
                lab = {"sin relleno, agrupacion 10 min": "Sets sin relleno (10 min)",
                       "agrupacion estricta 5 min, relleno 60 min": "Agrupación 5 min + relleno",
                       "sin relleno, agrupacion 5 min": "Sets sin relleno (5 min)"}.get(r["reconstruccion"], r["reconstruccion"])
                out.append((lab, r["OR M2 (IC95%)"], GRIS))
        return out

    fig, ax = plt.subplots(figsize=(8.4, 8.6))
    y, yt, yl = 0, [], []
    for sitio, coh, et in COHORTES[::-1]:
        for lab, txt, col in filas_sens(sitio, coh)[::-1]:
            o, (lo, hi) = num(txt), ic(txt)
            if np.isnan(o):
                continue
            ax.plot([lo, hi], [y, y], color=col, lw=1.3); ax.plot(o, y, "o", color=col, ms=4.8)
            ax.text(5.3, y, f"{es(f'{o:.2f}')} ({es(f'{lo:.2f}')}–{es(f'{hi:.2f}')})", va="center", fontsize=7, color=col)
            yt.append(y); yl.append(lab); y += 1
        ax.text(0.72, y - 0.2, f"{et} · {sitio}", fontsize=8.3, fontweight="bold", color=NEGRO)
        y += 1.1
    ax.axvline(1, color=NEGRO, lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.72, 5.2); ax.set_xticks([0.75, 1, 1.5, 2, 3, 4])
    ax.set_xticklabels(["0,75", "1", "1,5", "2", "3", "4"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.2); ax.set_ylim(-0.8, y)
    ax.set_xlabel("OR ajustado (modelo 2) de la normalización aparente frente a estables (escala log)")
    ax.set_title("La asociación se mantiene con definiciones, ajustes y reconstrucciones alternativas",
                 fontsize=10, loc="left", color=NEGRO)
    guardar_fig(fig, "Figura3_sensibilidad")
    for a in ("fase7a_contrastes.csv", "fase7a_frecuencia.csv", "fase7a_definiciones.csv", "fase7a_reconstruccion_mcmed.csv"):
        fuente("Figura 3", S7, a)

# ============================================================================
# FIGURA 4: calibracion en normalizacion aparente, N2 frente a N3
# ============================================================================
if c_cur is not None and c_des is not None:
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6))
    for j, (sitio, coh, et) in enumerate(COHORTES):
        for i, (ev, evlab) in enumerate(((VC, "VC agrupada"), (TEMP, "Partición temporal"))):
            ax = axes[i, j]
            cur = sel(c_cur, sitio, coh, evaluacion=ev, subgrupo="normalizacion aparente")
            if cur.empty:
                ax.axis("off"); continue
            mx = max(cur["predicho_%"].max(), cur["observado_%"].max()) * 1.15
            ax.plot([0, mx], [0, mx], color=GRIS_CL, lw=1)
            notas = []
            for m, col, mk in (("N2", GRIS, "s"), ("N3", AZUL, "o")):
                cc = cur[cur["modelo"] == m]
                ax.plot(cc["predicho_%"], cc["observado_%"], mk + "-", color=col, ms=4, lw=1, label=m)
                d = sel(c_des, sitio, coh, evaluacion=ev, subgrupo="normalizacion aparente", modelo=m)
                if not d.empty:
                    notas.append(f"{m}: {est_ic(d['CITL_relativo'].iloc[0], d['CITL_relativo_IC95'].iloc[0], 2)}")
            ax.set_xlim(0, mx); ax.set_ylim(0, mx)
            ax.text(0.03, 0.97, "CITL relativo\n" + "\n".join(notas), transform=ax.transAxes, va="top", fontsize=6.6, color=NEGRO)
            if i == 0:
                ax.set_title(f"{sitio}\n{et.split(' · ')[0]}", fontsize=8.6)
            if j == 0:
                ax.set_ylabel(f"{evlab}\nObservado (%)", fontsize=8.2)
            if i == 1:
                ax.set_xlabel("Predicho (%)", fontsize=8.2)
    axes[0, 0].legend(["Diagonal", "N2: clínico + NEWS2 final", "N3: + NEWS2 previo máximo"], frameon=False, fontsize=6.6, loc="upper right")
    fig.suptitle("Calibración en pacientes con normalización aparente: modelo sin gravedad previa (N2) frente a modelo "
                 "con gravedad previa (N3)\n(quintiles de riesgo predicho dentro del subgrupo; CITL relativo = CITL del "
                 "subgrupo menos CITL de toda la cohorte, IC95% bootstrap agrupado por paciente)", fontsize=9.3, x=0.01, ha="left")
    fig.tight_layout()
    guardar_fig(fig, "Figura4_calibracion_subgrupo")
    fuente("Figura 4", S7, "fase7c_curvas.csv"); fuente("Figura 4", S7, "fase7c_desempeno.csv")

# ============================================================================
# TABLAS PRINCIPALES
# ============================================================================
principales, supl = {}, {}
GNOM = {"Toda la cohorte": "Toda la cohorte", "normalizacion": "Normalización aparente",
        "estable": "Estable desde el triage", "ultimo_anormal": "Último set anormal"}

if c_t1 is not None:
    t = c_t1.copy()
    t["_o"] = t["grupo"].map({"Toda la cohorte": 0, "normalizacion": 1, "estable": 2, "ultimo_anormal": 3})
    t["_c"] = t.apply(lambda r: [(s, c) for s, c, _ in COHORTES].index((r["centro"], coh_key(r["cohorte"]))), axis=1)
    t = t.sort_values(["_c", "_o"])
    T1 = pd.DataFrame({"Centro": t["centro"], "Cohorte": t["cohorte"], "Grupo": t["grupo"].map(GNOM),
                       "Visitas (pacientes únicos)": [f"{miles(a)} ({miles(b)})" for a, b in zip(t["visitas"], t["pacientes_unicos"])],
                       "% de la cohorte": t["pct_visitas_cohorte"].map(es),
                       "Edad, mediana (RIC)": t["edad_mediana_RIC"].map(es), "Mujeres, %": t["mujeres_%"].map(es),
                       "ESI 1-2, %": t["ESI_1_2_%"].map(es), "Llegada en ambulancia, %": t["llegada_EMS_%"].map(es),
                       "Sets, mediana (RIC)": t["sets_mediana_RIC"].map(es),
                       "NEWS2 triage": t["NEWS2_triage_mediana_RIC"].map(es), "NEWS2 máximo": t["NEWS2_maximo_mediana_RIC"].map(es),
                       "NEWS2 final": t["NEWS2_final_mediana_RIC"].map(es), "Estadía (h)": t["estadia_h_mediana_RIC"].map(es),
                       "Desenlace, n (%)": [f"{miles(e)} ({es(p)})" for e, p in zip(t["eventos"], t["desenlace_%"])]})
    principales["Tabla1_caracteristicas"] = (T1, "Tabla 1. Características por centro, cohorte y grupo. Las unidades de análisis son "
                                             "visitas; entre paréntesis, pacientes únicos. NEWS2 modificado (sin oxígeno ni conciencia), "
                                             "mediana (RIC).")
    fuente("Tabla 1", S7, "fase7c_tabla1.csv")

if con is not None and c_t1 is not None:
    filas = []
    for sitio, coh, et in COHORTES:
        t = sel(c_t1, sitio, coh)
        g = lambda k: t[t["grupo"] == k].iloc[0]
        tot, an_, es_, ua = g("Toda la cohorte"), g("normalizacion"), g("estable"), g("ultimo_anormal")
        o = lambda cont, mod: sel(con, sitio, coh, contraste=cont, modelo=mod).iloc[0]
        p2, p1, p3 = (o("normalizacion vs estable", m) for m in ("M2 principal", "M1 sensibilidad", "M3 sensibilidad (proceso)"))
        q2 = o("normalizacion vs ultimo set anormal", "M2 principal")
        filas.append({"Cohorte": et, "Centro": sitio, "n analizado (eventos)": f"{miles(p2['n_analizado'])} ({miles(p2['eventos'])})",
                      "Normalización aparente, n (%)": f"{miles(an_['visitas'])} ({es(an_['pct_visitas_cohorte'])})",
                      "Desenlace normalización, %": es(f"{an_['desenlace_%']:.2f}"), "Desenlace estables, %": es(f"{es_['desenlace_%']:.2f}"),
                      "Desenlace último anormal, %": es(f"{ua['desenlace_%']:.2f}"),
                      "OR vs estables, modelo 2 (IC95%); p": f"{es(p2['OR (IC95%)'])}; {es(p2['p'])}",
                      "OR vs último set anormal, modelo 2 (IC95%); p": f"{es(q2['OR (IC95%)'])}; {es(q2['p'])}",
                      "Sensibilidad vs estables: modelo 1 / modelo 3": f"{es(p1['OR (IC95%)'])} / {es(p3['OR (IC95%)'])}"})
    principales["Tabla2_riesgo"] = (pd.DataFrame(filas), "Tabla 2. Tasas de desenlace y asociación ajustada. Modelo 2 (principal): edad, "
                                    "sexo, ESI, motivo de consulta (30 más frecuentes) y modo de llegada; modelo 1: edad, sexo y ESI; "
                                    "modelo 3: modelo 2 + proceso de atención (log n.º de sets, estadía, horas desde el último set). "
                                    "Regresión logística con errores estándar agrupados por paciente.")
    fuente("Tabla 2", S7, "fase7a_contrastes.csv"); fuente("Tabla 2", S7, "fase7c_tabla1.csv")

def dauroc(src, sitio, coh, ev, sg="toda la cohorte", cont="N3 - N2"):
    r = sel(src, sitio, coh, evaluacion=ev, subgrupo=sg, contraste=cont)
    return est_ic(r["dAUROC"].iloc[0], r["dAUROC_IC95"].iloc[0], 4) if not r.empty else ""


if b_des is not None and b_dif is not None:
    filas = []
    for sitio, coh, et in COHORTES:
        d = sel(b_des, sitio, coh, evaluacion=VC, subgrupo="toda la cohorte")
        au = lambda m: (lambda r: est_ic(r["AUROC"], r["AUROC_IC95"]))(d[d["modelo"] == m].iloc[0])
        row = {"Cohorte": et, "Centro": sitio, "AUROC N0: NEWS2 final": au("N0"), "AUROC N2: clínico + NEWS2 final": au("N2"),
               "AUROC N3: N2 + NEWS2 previo": au("N3"),
               "ΔAUROC N3−N2, VC": dauroc(c_dif if c_dif is not None else b_dif, sitio, coh, VC),
               "ΔAUROC N3−N2, temporal": dauroc(c_dif if c_dif is not None else b_dif, sitio, coh, TEMP)}
        if c_des is not None:
            r = sel(c_des, sitio, coh, evaluacion=TEMP, subgrupo="normalizacion aparente", modelo="N2")
            row["Normalización en prueba temporal: n (eventos)"] = f"{miles(r['n'].iloc[0])} ({int(r['eventos'].iloc[0])})" if not r.empty else ""
        if c_dif is not None:
            row["Sensibilidad ΔAUROC N4−N2P (+ proceso), VC"] = dauroc(c_dif, sitio, coh, VC, cont="N4 - N2P")
            row["Sensibilidad ΔAUROC N4−N2P, temporal"] = dauroc(c_dif, sitio, coh, TEMP, cont="N4 - N2P")
        if c_des is not None:
            for ev, nom in ((VC, "VC"), (TEMP, "temporal")):
                v = []
                for m in ("N2", "N3"):
                    r = sel(c_des, sitio, coh, evaluacion=ev, subgrupo="normalizacion aparente", modelo=m)
                    v.append(est_ic(r["CITL_relativo"].iloc[0], r["CITL_relativo_IC95"].iloc[0], 2) if not r.empty else "")
                row[f"CITL relativo en normalización N2 → N3, {nom}"] = " → ".join(v)
        if b_rec is not None:
            r = sel(b_rec, sitio, coh).sort_values("umbral_%").iloc[0]
            row["Beneficio neto N3−N2 por 1.000 (umbral)"] = (f"{es(round(r['beneficio_neto_N3_x1000'] - r['beneficio_neto_N2_x1000'], 2))} "
                                                              f"({es(r['umbral_%'])}%)")
        filas.append(row)
    principales["Tabla3_valor_incremental"] = (
        pd.DataFrame(filas),
        "Tabla 3. Valor incremental del NEWS2 previo máximo. Regresión logística; VC: validación cruzada de 5 particiones agrupadas "
        "por paciente; temporal: entrenamiento en años previos y prueba en el periodo final sin pacientes compartidos. IC95% por "
        "bootstrap agrupado por paciente (1.000 réplicas, semilla 20261103). CITL relativo > 0 indica riesgo subestimado en el "
        "subgrupo con normalización aparente respecto de la cohorte completa. N4−N2P: mismo contraste añadiendo el proceso de "
        "atención a ambos modelos. Las estimaciones temporales del subgrupo en MC-MED se basan en 9 y 18 eventos y son imprecisas. "
        "ΔAUROC N3−N2 recalculado en la muestra común de la fase 7c (estimaciones puntuales idénticas a la fase 7b).")
    for a in ("fase7b_desempeno.csv", "fase7b_diferencias.csv", "fase7b_reclasificacion.csv", "fase7c_diferencias.csv", "fase7c_desempeno.csv"):
        fuente("Tabla 3", S7, a)

# ============================================================================
# SUPLEMENTO
# ============================================================================
def supl_add(clave, df, titulo, archivo):
    if df is None or df.empty:
        return
    supl[clave] = (df.apply(lambda col: col.map(es)), titulo)
    fuente(clave.split("_")[0], S7, archivo)


supl_add("S1_fenotipos", fen, "Tabla S1. Fenotipos preespecificados dentro de la normalización aparente (OR ajustado, modelo 2; "
         "asociativo, no causal).", "fase7a_fenotipos.csv")
if fre is not None and b_int is not None:
    supl_add("S2_frecuencia", fre[~fre["analisis"].str.startswith("test de interaccion")],
             "Tabla S2. Análisis por número de controles de signos vitales y definición basada solo en el triage.", "fase7a_frecuencia.csv")
    supl_add("S2b_interaccion", b_int, "Tabla S2b. Test de interacción grupo × número de controles (estratos ≤4 / 5-6 / ≥7; "
             "sustituye al de 4 estratos, que tenía un estrato sin eventos).", "fase7b_interaccion.csv")
supl_add("S3_definiciones", dfs, "Tabla S3. Definiciones alternativas de normalización aparente (modelo 2).", "fase7a_definiciones.csv")
supl_add("S3b_reconstruccion", rec, "Tabla S3b. MC-MED: reconstrucción alternativa de los sets de signos vitales (modelo 2).",
         "fase7a_reconstruccion_mcmed.csv")
supl_add("S4_N0_N3", b_des, "Tabla S4. Desempeño completo de N0-N3 (toda la cohorte, subgrupo y partición temporal).", "fase7b_desempeno.csv")
supl_add("S5_diferencias", b_dif, "Tabla S5. Diferencias pareadas de AUROC y AUPRC.", "fase7b_diferencias.csv")
supl_add("S6_reclasificacion", b_rec, "Tabla S6. Reclasificación y beneficio neto, N3 frente a N2, umbrales preespecificados.",
         "fase7b_reclasificacion.csv")
supl_add("S7_calibracion_N4", c_des, "Tabla S7. Calibración (pendiente, CITL y CITL relativo, IC95%) de N2, N3, N2P y N4, en VC "
         "agrupada y partición temporal.", "fase7c_desempeno.csv")
supl_add("S7b_contrastes_N4", c_dif, "Tabla S7b. Contrastes N3−N2, N4−N2P y N4−N3 en la muestra común.", "fase7c_diferencias.csv")
if b_exc is not None and fal is not None:
    supl_add("S8_faltantes", fal, "Tabla S8. Datos faltantes por variable.", "fase7a_faltantes.csv")
    supl_add("S8b_exclusiones", b_exc, "Tabla S8b. Denominadores y exclusiones por datos faltantes (análisis de modelos).",
             "fase7b_exclusiones.csv")

# ============================================================================
# EXPORTAR
# ============================================================================
def a_word(tabs, ruta, encabezado):
    doc = docx.Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    for m in ("left_margin", "right_margin"):
        setattr(sec, m, docx.shared.Cm(1.5))
    doc.add_heading(encabezado, level=1)
    for nombre, (df, titulo) in tabs.items():
        p = doc.add_paragraph(); r = p.add_run(titulo); r.bold = True; r.font.size = Pt(9.5)
        tb = doc.add_table(rows=1, cols=len(df.columns)); tb.style = "Light Grid Accent 1"
        for j, c in enumerate(df.columns):
            tb.rows[0].cells[j].text = str(c)
        for _, rr in df.iterrows():
            cells = tb.add_row().cells
            for j, c in enumerate(df.columns):
                cells[j].text = str(rr[c])
        for row in tb.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    for rn in par.runs:
                        rn.font.size = Pt(7)
        doc.add_paragraph()
    doc.save(ruta)


if principales:
    a_word(principales, os.path.join(DEST, "Tablas_principales.docx"), "Artículo 1 · Tablas principales")
if supl:
    a_word(supl, os.path.join(DEST, "Tablas_suplementarias.docx"), "Artículo 1 · Material suplementario")
tz = pd.DataFrame(traza).drop_duplicates()
with pd.ExcelWriter(os.path.join(DEST, "Tablas_articulo1_v2.xlsx")) as xw:
    for nombre, (df, _) in {**principales, **supl}.items():
        df.to_excel(xw, sheet_name=nombre[:31], index=False)
        df.to_csv(os.path.join(DEST, f"{nombre}.csv"), index=False)
    tz.to_excel(xw, sheet_name="trazabilidad", index=False)
tz.to_csv(os.path.join(DEST, "trazabilidad.csv"), index=False)
hechos += list(principales)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 60)
for nombre, (df, titulo) in principales.items():
    if nombre != "Tabla1_caracteristicas":
        print(f"\n=== {titulo[:90]}... ===")
        print(df.to_string(index=False))
print(f"\nTexto principal: {len(hechos)} de 7 piezas · suplemento: {len(supl)} tablas")
print("Carpeta:", DEST)
print("Archivos:", sorted(os.listdir(DEST)))
