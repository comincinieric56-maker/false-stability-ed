# ============================================================================
# FALSE-STABILITY - Figuras 1 a 4 en ingles para AEM (600 dpi), desde los
# resultados definitivos (desenlace A corregido). CPU, <1 min.
# Salida: .../False Stability/AEM_envio_final/Figure1-4 (.tif 600 dpi y .png)
# ============================================================================
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass
FOLDER = globals().get("FOLDER", "/content/drive/MyDrive/ARTICULOS MIOS/False Stability")
S_MI, S_MC, S7 = (os.path.join(FOLDER, x) for x in ("salida", "salida_mcmed", "salida_fase7"))
DEST = os.path.join(FOLDER, "AEM_envio_final")
os.makedirs(DEST, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GRAY, DARK, LIGHT = "#1f4e9c", "#6b7280", "#111827", "#d1d5db"


def save(fig, name):
    fig.savefig(os.path.join(DEST, name + ".tif"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(DEST, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  listo:", name)


def ckey(s):
    s = str(s).strip().upper()
    return "A" if s.startswith("A") else ("B*" if s.startswith("B*") else "B")


EN = [(r"visitas a urgencia", "ED visits"), (r"alta a domicilio", "Discharged home"), (r"edad", "Age ≥18 years"),
      (r"visitas con hospitalizacion", "ED visits with admission"), (r"primera unidad", "First unit: general ward"),
      (r"alta desde urgencia", "Discharged from ED"), (r"sin fuga", "Excluding elopement, AMA, and ED death"),
      (r"sala u observacion", "Ward or observation admission"), (r"sets", "≥2 valid vital-sign sets")]


def en(p):
    for pat, lab in EN:
        if re.search(pat, str(p).lower()):
            return lab
    return str(p)


# ---------------- Figures 2 and 3: adjusted associations and sensitivity ----------------
from matplotlib.ticker import FixedLocator, NullLocator
BLUE2,GRAY2,DARK2="#1f4e9c","#6b7280","#111827"
coh=["MIMIC-IV-ED, cohort A (discharge)","MC-MED, cohort A (discharge)","MIMIC-IV-ED, cohort B (ward)","MC-MED, cohort B* (ward/observation)"]
def logax(ax,lo,hi,ticks):
    ax.set_xscale("log"); ax.set_xlim(lo,hi); ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([("%g"%t) for t in ticks]); ax.xaxis.set_minor_locator(NullLocator()); ax.axvline(1,color=DARK,lw=0.7,ls="--")
# Figure 2
d={0:[("Model 2 vs stable (primary)",1.48,1.34,1.64,BLUE,"o",True),("Model 1 vs stable",1.64,1.49,1.81,BLUE,"o",False),("Model 3 vs stable",1.39,1.26,1.55,BLUE,"o",False),("Model 2 vs abnormal final set",0.91,0.80,1.03,GRAY,"s",True)],
   1:[("Model 2 vs stable (primary)",1.61,1.27,2.03,BLUE,"o",True),("Model 1 vs stable",1.70,1.35,2.15,BLUE,"o",False),("Model 3 vs stable",1.40,1.09,1.79,BLUE,"o",False),("Model 2 vs abnormal final set",0.89,0.69,1.15,GRAY,"s",True)],
   2:[("Model 2 vs stable (primary)",1.48,1.35,1.63,BLUE,"o",True),("Model 1 vs stable",1.52,1.39,1.67,BLUE,"o",False),("Model 3 vs stable",1.45,1.32,1.59,BLUE,"o",False),("Model 2 vs abnormal final set",0.46,0.42,0.50,GRAY,"s",True)],
   3:[("Model 2 vs stable (primary)",1.95,1.60,2.38,BLUE,"o",True),("Model 1 vs stable",2.09,1.72,2.55,BLUE,"o",False),("Model 3 vs stable",1.66,1.35,2.04,BLUE,"o",False),("Model 2 vs abnormal final set",0.56,0.47,0.66,GRAY,"s",True)]}
fig,ax=plt.subplots(figsize=(6.97,5.2)); y=0; yt=[]; yl=[]
for k in [3,2,1,0]:
    for lab,o,lo,hi,c,m,f in d[k][::-1]:
        ax.plot([lo,hi],[y,y],color=c,lw=1.4 if f else 0.9); ax.plot(o,y,m,color=c,mfc=c if f else "white",ms=5)
        ax.text(3.35,y,f"{o:.2f} ({lo:.2f}–{hi:.2f})",va="center",fontsize=7.2,color=c); yt.append(y); yl.append(lab); y+=1
    ax.text(0.36,y-0.15,coh[k],fontsize=7.6,fontweight="bold",color=DARK); y+=1.0
logax(ax,0.35,3.3,[0.4,0.5,0.75,1,1.5,2,3]); ax.set_yticks(yt); ax.set_yticklabels(yl,fontsize=7.2); ax.set_ylim(-0.8,y-0.3)
ax.text(3.35,y-0.55,"OR (95% CI)",fontsize=7.2,fontweight="bold"); ax.set_xlabel("Adjusted odds ratio for apparent normalization (log scale)")
fig.savefig(os.path.join(DEST,"Figure2.tif"),dpi=600,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"}); fig.savefig(os.path.join(DEST,"Figure2.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
# Figure 3
rows=["Primary analysis","+ adjustment for number of sets","Triage-only definition","Prior NEWS2 ≥5","Final NEWS2 = 0","Complete 5-parameter sets only","Excluding supplemental oxygen","No carry-forward, 10-min sets","5-min sets + 60-min carry-forward","No carry-forward, 5-min sets","Fixed 10-min windows"]
v={0:[(1.48,1.34,1.64),(1.42,1.27,1.57),(1.56,1.36,1.78),(1.54,1.19,1.99),(1.48,1.29,1.70),(1.58,1.41,1.77)],
   1:[(1.61,1.27,2.03),(1.46,1.13,1.88),(1.64,1.24,2.16),(1.50,0.80,2.80),(1.57,1.12,2.20),(1.86,1.36,2.55),(1.60,1.25,2.04),(1.72,1.34,2.19),(1.61,1.27,2.04),(1.69,1.32,2.17),(1.61,1.28,2.04)],
   2:[(1.48,1.35,1.63),(1.61,1.46,1.78),(1.43,1.26,1.62),(2.06,1.76,2.43),(1.46,1.27,1.68),(1.34,1.19,1.51)],
   3:[(1.95,1.60,2.38),(1.91,1.53,2.39),(1.84,1.45,2.34),(2.77,2.02,3.80),(1.85,1.34,2.56),(1.72,1.34,2.21),(1.78,1.39,2.29),(1.91,1.56,2.34),(1.98,1.62,2.42),(1.99,1.63,2.44),(1.97,1.61,2.40)]}
fig,ax=plt.subplots(figsize=(6.97,8.6)); y=0; yt=[]; yl=[]
for k in [3,2,1,0]:
    for i in range(len(v[k]))[::-1]:
        if v[k][i] is None: continue
        o,lo,hi=v[k][i]; c=BLUE if i==0 else GRAY
        ax.plot([lo,hi],[y,y],color=c,lw=1.3); ax.plot(o,y,"o",color=c,ms=4.5)
        ax.text(4.25,y,f"{o:.2f} ({lo:.2f}–{hi:.2f})",va="center",fontsize=7,color=c); yt.append(y); yl.append(rows[i]); y+=1
    ax.text(0.72,y-0.15,coh[k],fontsize=7.6,fontweight="bold",color=DARK); y+=1.0
logax(ax,0.7,4.2,[0.75,1,1.5,2,3,4]); ax.set_yticks(yt); ax.set_yticklabels(yl,fontsize=7); ax.set_ylim(-0.8,y-0.3)
ax.text(4.25,y-0.55,"OR (95% CI)",fontsize=7,fontweight="bold"); ax.set_xlabel("Adjusted odds ratio (model 2), apparent normalization vs stable from triage (log scale)")
fig.savefig(os.path.join(DEST,"Figure3.tif"),dpi=600,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"}); fig.savefig(os.path.join(DEST,"Figure3.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("  listo: Figure2, Figure3")

# ---------------- Figure 1: selection flow ----------------
flu = {"MIMIC-IV-ED": pd.read_csv(os.path.join(S_MI, "flujo_seleccion.csv")),
       "MC-MED": pd.read_csv(os.path.join(S_MC, "fase6b_flujo.csv"))}
panels = [("MIMIC-IV-ED", "A", "Cohort A: discharge"), ("MIMIC-IV-ED", "B", "Cohort B: ward"),
          ("MC-MED", "A", "Cohort A: discharge"), ("MC-MED", "B*", "Cohort B*: ward/observation")]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 3.0))
for ax, (site, k, lab) in zip(axes, panels):
    d = flu[site]
    d = d[d["cohorte"].map(ckey).isin([k] if k != "B*" else ["B", "B*"])].reset_index(drop=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(f"{site}\n{lab}", fontsize=7.5, fontweight="bold", color=DARK)
    n = len(d); ys = np.linspace(0.86, 0.12, max(n, 2))[:n]
    for i, r in d.iterrows():
        last = i == n - 1
        ax.add_patch(FancyBboxPatch((0.04, ys[i] - 0.075), 0.92, 0.15, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc="#e8eef8" if last else "white", ec=BLUE if last else GRAY, lw=0.8))
        ax.text(0.5, ys[i] + 0.028, en(r["paso"]), ha="center", va="center", fontsize=5.8, color=DARK)
        ax.text(0.5, ys[i] - 0.032, f"n = {int(r['n']):,}", ha="center", va="center", fontsize=7, fontweight="bold", color=DARK)
        if not last:
            ax.annotate("", xy=(0.5, ys[i + 1] + 0.078), xytext=(0.5, ys[i] - 0.078), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
fig.tight_layout()
save(fig, "Figure1")

# ---------------- Figure 4: calibration in apparent normalization ----------------
cur = pd.read_csv(os.path.join(S7, "fase7c_curvas.csv"))
des = pd.read_csv(os.path.join(S7, "fase7c_desempeno.csv"))
VC, TE = "VC 5 particiones por paciente", "particion temporal (sin pacientes compartidos)"
cols = [("MIMIC-IV-ED", "A", "MIMIC-IV-ED · cohort A"), ("MC-MED", "A", "MC-MED · cohort A"),
        ("MIMIC-IV-ED", "B", "MIMIC-IV-ED · cohort B"), ("MC-MED", "B*", "MC-MED · cohort B*")]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9))
for j, (site, k, title) in enumerate(cols):
    for i, (ev, rowlab) in enumerate(((VC, "Cross-validation"), (TE, "Temporal test"))):
        ax = axes[i, j]
        c = cur[(cur["centro"] == site) & (cur["cohorte"].map(ckey) == k) & (cur["evaluacion"] == ev) & (cur["subgrupo"] == "normalizacion aparente")]
        mx = max(c["predicho_%"].max(), c["observado_%"].max()) * 1.12
        ax.plot([0, mx], [0, mx], color=LIGHT, lw=0.8)
        note = []
        for m, col, mk, lab in (("N2", GRAY, "s", "N2: clinical + final NEWS2"), ("N3", BLUE, "o", "N3: + prior maximum NEWS2")):
            cc = c[c["modelo"] == m]
            ax.plot(cc["predicho_%"], cc["observado_%"], mk + "-", color=col, ms=2.8, lw=0.9, label=lab)
            r = des[(des["centro"] == site) & (des["cohorte"].map(ckey) == k) & (des["evaluacion"] == ev)
                    & (des["subgrupo"] == "normalizacion aparente") & (des["modelo"] == m)]
            if not r.empty:
                note.append(f"{m} {r['CITL_relativo'].iloc[0]:.2f}")
        ev_n = int(c[c["modelo"] == "N2"]["eventos"].sum())
        ax.text(0.04, 0.96, "Relative CITL\n" + "\n".join(note) + f"\n{ev_n} events", transform=ax.transAxes, va="top", fontsize=5.6)
        ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.tick_params(labelsize=6)
        if i == 0:
            ax.set_title(title, fontsize=6.8, fontweight="bold")
        if j == 0:
            ax.set_ylabel(f"{rowlab}\nObserved risk (%)", fontsize=6.5)
        if i == 1:
            ax.set_xlabel("Predicted risk (%)", fontsize=6.5)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "Figure4")
print("Carpeta:", DEST)
