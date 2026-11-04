# ============================================================================
# FALSE-STABILITY - Fase 5: red de secuencia M3 (GPU)
#   Rama de secuencia: GRU sobre cada set de signos vitales (valores, puntajes
#   NEWS2, mascaras de faltantes y tiempos irregulares).
#   Rama tabular: FT-Transformer con estaticas, historia y tratamiento.
#   Fusion -> riesgo. Comparacion contra M2 (gradient boosting de la fase 3).
#
# Requiere: celda de la fase 3 corrida en esta sesion, y el archivo
# /content/false_stability_cohort.py (lo crea la celda %%writefile de la fase 2).
# Entorno de ejecucion -> Cambiar tipo de entorno -> GPU (T4 basta).
# Todo lo que imprime y los graficos son agregados y se pueden compartir.
# ============================================================================
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for _need in ("prep", "fit_gbm", "boot_ci", "FEATS", "load_out", "read_from_zip", "ed_all",
              "STATIC_NUM", "TX_NUM", "CAT", "metrics"):
    if _need not in globals():
        raise RuntimeError("Primero corre la celda de la fase 3 en esta sesion.")
sys.path.insert(0, "/content")
try:
    from false_stability_cohort import build_vitals_long
except ImportError:
    raise RuntimeError("Falta /content/false_stability_cohort.py: corre la celda %%writefile de la fase 2.")

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
L_MAX = 24            # ultimos 24 sets por visita (el p95 esta muy por debajo)
D = 64                # dimension oculta
N_SEEDS = 3           # semillas del modelo completo (se promedian)
EPOCHS = 40
PATIENCE = 6
BATCH = 1024
LR = 1e-3
COHORTES = [("cohorte_B", "B: UCI 48h", True),          # (archivo, nombre, correr ablaciones)
            ("cohorte_A", "A: reconsulta 72h", False)]
SEQ_ARCH = "gru"      # "gru" o "transformer"

SEQ_VITALS = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature"]
SEQ_SCORES = ["s_rr", "s_spo2", "s_sbp", "s_hr", "s_temp", "news2"]
SEQ_TIME = ["t_arr", "t_exit", "dt_prev"]
SEQ_CONT = SEQ_VITALS + SEQ_SCORES + SEQ_TIME
SEQ_FLAGS = [v + "_m" for v in SEQ_VITALS] + ["is_triage"]
SEQ_CH = SEQ_CONT + SEQ_FLAGS
TIME_IDX = [SEQ_CH.index(c) for c in SEQ_TIME]
TAB_NUM = STATIC_NUM + TX_NUM


# ----------------------------------------------------------------------------
# Secuencias
# ----------------------------------------------------------------------------
def build_long_seq():
    print("Reconstruyendo secuencias de signos vitales...")
    tri = read_from_zip("triage")
    vs = read_from_zip("vitalsign", parse_dates=["charttime"])
    long = build_vitals_long(ed_all, tri, vs)
    h = lambda td: td.dt.total_seconds() / 3600
    long["t_arr"] = np.log1p(h(long["charttime"] - long["intime"]).clip(lower=0))
    long["t_exit"] = np.log1p(h(long["outtime"] - long["charttime"]).clip(lower=0))
    long["dt_prev"] = np.log1p(h(long.groupby("stay_id")["charttime"].diff()).fillna(0).clip(lower=0))
    long["is_triage"] = (long["source"] == "triage").astype(float)
    for v in SEQ_VITALS:
        long[v + "_m"] = long[v].isna().astype(float)
    long = long.sort_values(["stay_id", "charttime"]).groupby("stay_id").tail(L_MAX)
    return long[["stay_id"] + SEQ_CH]


def seq_stats(long, stay_ids):
    sub = long[long["stay_id"].isin(stay_ids)]
    return sub[SEQ_CONT].mean(), sub[SEQ_CONT].std().replace(0, 1)


def build_seq(c, long, stats):
    """Devuelve X [N, L_MAX, F] (alineado a la izquierda) y largos [N], en el orden de c."""
    mu, sd = stats
    sub = long[long["stay_id"].isin(c["stay_id"])].copy()
    sub[SEQ_CONT] = ((sub[SEQ_CONT] - mu) / sd).fillna(0)
    row = pd.Categorical(sub["stay_id"], categories=c["stay_id"].to_numpy()).codes
    pos = sub.groupby("stay_id").cumcount().to_numpy()
    X = np.zeros((len(c), L_MAX, len(SEQ_CH)), dtype=np.float32)
    X[row, pos, :] = sub[SEQ_CH].to_numpy(np.float32)
    lengths = np.bincount(row, minlength=len(c)).astype(np.int64)
    return X, np.maximum(lengths, 1)


def tab_encoders(tr):
    med = tr[TAB_NUM].median()
    mu = tr[TAB_NUM].fillna(med).mean()
    sd = tr[TAB_NUM].fillna(med).std().replace(0, 1)
    cats = {c: {v: i + 1 for i, v in enumerate(tr[c].astype(str).unique())} for c in CAT}  # 0 = desconocido
    return med, mu, sd, cats


def build_tab(c, enc):
    med, mu, sd, cats = enc
    xn = ((c[TAB_NUM].fillna(med) - mu) / sd).to_numpy(np.float32)
    xc = np.stack([c[k].astype(str).map(cats[k]).fillna(0).to_numpy(np.int64) for k in CAT], 1)
    return xn, xc


def split_val(tr, frac=0.15, seed=SEED):
    subj = tr["subject_id"].unique()
    val_subj = set(np.random.default_rng(seed).choice(subj, int(frac * len(subj)), replace=False))
    return tr["subject_id"].isin(val_subj).to_numpy()


# ============================================================================
# MODELO
# ============================================================================
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "(sin GPU: sera lento)")


class FTTabular(nn.Module):
    """FT-Transformer liviano: un token por variable + CLS."""
    def __init__(self, n_num, cat_cards, d=D, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.embs = nn.ModuleList([nn.Embedding(k, d) for k in cat_cards])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, xn, xc):
        tn = xn.unsqueeze(-1) * self.w + self.b
        tc = torch.stack([e(xc[:, i]) for i, e in enumerate(self.embs)], 1)
        t = torch.cat([self.cls.expand(xn.shape[0], -1, -1), tn, tc], 1)
        return self.norm(self.enc(t)[:, 0])


class SeqEncoder(nn.Module):
    def __init__(self, f, d=D, arch=SEQ_ARCH, drop=0.1):
        super().__init__()
        self.arch = arch
        self.inp = nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Dropout(drop))
        if arch == "gru":
            self.rnn = nn.GRU(d, d, num_layers=1, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, lengths):
        h = self.inp(x)
        if self.arch == "gru":
            packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hn = self.rnn(packed)
            return self.norm(hn[-1])
        mask = torch.arange(x.shape[1], device=x.device)[None, :] >= lengths[:, None]
        z = self.enc(h, src_key_padding_mask=mask)
        last = z[torch.arange(len(z), device=z.device), lengths - 1]
        mean = (z * (~mask).unsqueeze(-1)).sum(1) / lengths[:, None]
        return self.norm(last + mean)


class M3(nn.Module):
    def __init__(self, f_seq, n_num, cat_cards, use_seq=True, use_tab=True, d=D, drop=0.2):
        super().__init__()
        self.use_seq, self.use_tab = use_seq, use_tab
        self.seq = SeqEncoder(f_seq, d) if use_seq else None
        self.tab = FTTabular(n_num, cat_cards, d) if use_tab else None
        k = d * (int(use_seq) + int(use_tab))
        self.head = nn.Sequential(nn.Linear(k, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, xs, ln, xn, xc):
        parts = []
        if self.use_seq:
            parts.append(self.seq(xs, ln))
        if self.use_tab:
            parts.append(self.tab(xn, xc))
        return self.head(torch.cat(parts, 1)).squeeze(-1)


def to_dev(*arrs):
    return [torch.as_tensor(a).to(DEVICE) for a in arrs]


@torch.no_grad()
def predict(model, T, bs=4096):
    model.eval()
    xs, ln, xn, xc = T
    out = []
    for i in range(0, len(ln), bs):
        out.append(torch.sigmoid(model(xs[i:i + bs], ln[i:i + bs], xn[i:i + bs], xc[i:i + bs])).float().cpu())
    return torch.cat(out).numpy()


def train_one(Tfit, yfit, Tval, yval, cards, seed, use_seq=True, use_tab=True, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    model = M3(len(SEQ_CH), len(TAB_NUM), cards, use_seq, use_tab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.as_tensor(yfit, dtype=torch.float32, device=DEVICE)
    n = len(yfit)
    best, best_state, bad = -1, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            logit = model(Tfit[0][idx], Tfit[1][idx], Tfit[2][idx], Tfit[3][idx])
            loss = lossf(logit, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ap = average_precision_score(yval, predict(model, Tval))
        sched.step(ap)
        if ap > best + 1e-4:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    print(f"    {tag} semilla {seed}: {ep + 1} epocas, AUPRC val {best:.4f}, {time.time() - t0:.0f}s")
    return model


def eval_rows(y, preds, nombre, subset, ref="M2_gbm"):
    ci = boot_ci(y, preds, ref=ref)
    rows = []
    for k, p in preds.items():
        m = metrics(y, p)
        rows.append({"cohorte": nombre, "subgrupo": subset, "modelo": k, "n": len(y), "eventos": int(y.sum()),
                     "AUROC": round(m["AUROC"], 3), "AUROC_IC95": ci[k]["AUROC_ic"],
                     "AUPRC": round(m["AUPRC"], 3), "AUPRC_IC95": ci[k]["AUPRC_ic"],
                     "O/E": round(m["O/E"], 2), "pendiente_cal": round(m["pendiente"], 2),
                     f"dAUPRC_vs_{ref}": ci[k].get(f"dAUPRC_vs_{ref}", "")})
    return rows


def temporal_attribution(model, T, mask_rows, y, nombre):
    """Gradiente x entrada sobre la secuencia: que posicion (desde el final) pesa mas."""
    model.eval()
    xs, ln, xn, xc = T
    idx = torch.as_tensor(np.where(mask_rows)[0], device=DEVICE)
    pos_share, ch_share = [], []
    for i in range(0, len(idx), 2048):
        b = idx[i:i + 2048]
        x = xs[b].clone().requires_grad_(True)
        model.zero_grad()
        # cuDNN no permite backward de una GRU en modo eval: se desactiva solo aqui
        with torch.backends.cudnn.flags(enabled=False):
            model(x, ln[b], xn[b], xc[b]).sum().backward()
        a = (x.grad * x).abs().detach()                      # [B, L, F]
        per_pos = a.sum(-1)
        per_pos = per_pos / per_pos.sum(1, keepdim=True).clamp_min(1e-12)
        pos_share.append(per_pos.cpu().numpy())
        per_ch = a.sum(1)
        ch_share.append((per_ch / per_ch.sum(1, keepdim=True).clamp_min(1e-12)).cpu().numpy())
    P = np.concatenate(pos_share)
    C = np.concatenate(ch_share)
    L = ln[idx].cpu().numpy()
    yy = y[mask_rows]

    labels = []
    for l in L:
        lab = np.array(["intermedios"] * L_MAX, dtype=object)
        lab[l - 1] = "ultimo set"
        if l >= 2: lab[l - 2] = "penultimo"
        if l >= 3: lab[l - 3] = "antepenultimo"
        lab[0] = "triage / primer set" if l > 1 else "ultimo set"
        lab[l:] = "padding"
        labels.append(lab)
    labels = np.stack(labels)
    order = ["ultimo set", "penultimo", "antepenultimo", "intermedios", "triage / primer set"]
    rows = []
    for grp, sel in (("todos", np.ones(len(yy), bool)), ("con desenlace", yy == 1), ("sin desenlace", yy == 0)):
        r = {"cohorte": nombre, "grupo": grp, "n": int(sel.sum())}
        for o in order:
            r[o] = round(100 * (P[sel] * (labels[sel] == o)).sum(1).mean(), 1)
        rows.append(r)
    pos = pd.DataFrame(rows)

    groups = {"valores vitales": SEQ_VITALS, "puntajes NEWS2": SEQ_SCORES, "tiempos": SEQ_TIME,
              "faltantes/triage": SEQ_FLAGS}
    crow = []
    for grp, sel in (("con desenlace", yy == 1), ("sin desenlace", yy == 0)):
        r = {"cohorte": nombre, "grupo": grp}
        for g, chs in groups.items():
            r[g] = round(100 * C[sel][:, [SEQ_CH.index(c) for c in chs]].sum(1).mean(), 1)
        crow.append(r)
    return pos, pd.DataFrame(crow)


# ----------------------------------------------------------------------------
# Ejecucion
# ----------------------------------------------------------------------------
long_seq = build_long_seq()
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
rows, pos_all, ch_all = [], [], []

for fname, nombre, do_abl in COHORTES:
    print(f"\n### {nombre}")
    c = prep(load_out(fname)).reset_index(drop=True)
    tr, te = c[c["split"] == "train"].reset_index(drop=True), c[c["split"] == "test"].reset_index(drop=True)
    vmask = split_val(tr)
    fit, val = tr[~vmask].reset_index(drop=True), tr[vmask].reset_index(drop=True)

    stats = seq_stats(long_seq, fit["stay_id"])
    enc = tab_encoders(fit)
    cards = [len(enc[3][k]) + 1 for k in CAT]

    def tensors(d):
        X, ln = build_seq(d, long_seq, stats)
        xn, xc = build_tab(d, enc)
        return to_dev(X, ln, xn, xc)

    Tfit, Tval, Tte = tensors(fit), tensors(val), tensors(te)
    yfit, yval, yte = fit["y"].to_numpy(), val["y"].to_numpy(), te["y"].to_numpy()
    print(f"  fit {len(fit):,} / val {len(val):,} / prueba {len(te):,} | eventos prueba {yte.sum():,}")

    print("  M2 gradient boosting (referencia)...")
    preds = {"M2_gbm": fit_gbm(tr, FEATS["M2"])(te)}
    # Referencia solo fisiologica: M2 sin variables del proceso de atencion (fase 4)
    PROC = ["n_sets", "max_gap_h", "hours_last_set_to_exit", "ed_los_h", "sets_per_hour"]
    preds["M2_gbm sin proceso"] = fit_gbm(
        tr, dict(num=[f for f in FEATS["M2"]["num"] if f not in PROC], cat=CAT))(te)

    print("  M3 completo:")
    models = [train_one(Tfit, yfit, Tval, yval, cards, s, tag="M3") for s in range(N_SEEDS)]
    preds["M3"] = np.mean([predict(m, Tte) for m in models], 0)
    preds["M2+M3 promedio"] = (preds["M2_gbm"] + preds["M3"]) / 2

    if do_abl:
        print("  Ablaciones M3:")
        notime = lambda T: [T[0].clone().index_fill_(2, torch.as_tensor(TIME_IDX, device=DEVICE), 0)] + T[1:]
        m = train_one(notime(Tfit), yfit, notime(Tval), yval, cards, 0, tag="M3 sin tiempos")
        preds["M3 sin tiempos"] = predict(m, notime(Tte))
        m = train_one(Tfit, yfit, Tval, yval, cards, 0, use_tab=False, tag="M3 solo secuencia")
        preds["M3 solo secuencia"] = predict(m, Tte)
        m = train_one(Tfit, yfit, Tval, yval, cards, 0, use_seq=False, tag="M3 solo tabular")
        preds["M3 solo tabular"] = predict(m, Tte)

    an = te["apparent_normalization"].to_numpy()
    rows += eval_rows(yte, preds, nombre, "toda la prueba")
    rows += eval_rows(yte[an], {k: p[an] for k, p in preds.items()}, nombre, "normalizacion aparente")

    pos, ch = temporal_attribution(models[0], Tte, an, yte, nombre)
    pos_all.append(pos); ch_all.append(ch)
    for i, mdl in enumerate(models):
        torch.save(mdl.state_dict(), os.path.join(OUT, f"m3_{fname}_seed{i}.pt"))
    del Tfit, Tval, Tte
    torch.cuda.empty_cache()

res = pd.DataFrame(rows)
pos = pd.concat(pos_all)
ch = pd.concat(ch_all)

fig, ax = plt.subplots(figsize=(9, 4))
cols = ["ultimo set", "penultimo", "antepenultimo", "intermedios", "triage / primer set"]
sub = pos[pos["grupo"] != "todos"]
xx = np.arange(len(cols)); w = 0.8 / len(sub)
for j, (_, r) in enumerate(sub.iterrows()):
    ax.bar(xx + j * w, [r[c] for c in cols], w, label=f"{r['cohorte']} - {r['grupo']}")
ax.set_xticks(xx + w * (len(sub) - 1) / 2, cols)
ax.set(ylabel="% de la atribucion de la secuencia",
       title="M3 en normalizacion aparente: peso de cada momento de la trayectoria")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fase5_atribucion_temporal.png"), dpi=150)
plt.show()

for titulo, df, f in (("Desempeno M3 vs M2 (prueba temporal, IC95 bootstrap)", res, "fase5_desempeno"),
                      ("Atribucion por momento de la secuencia (%), normalizacion aparente", pos, "fase5_atrib_posicion"),
                      ("Atribucion por tipo de canal (%), normalizacion aparente", ch, "fase5_atrib_canal")):
    print(f"\n=== {titulo} ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, f + ".csv"), index=False)
