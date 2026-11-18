"""
Gera gsvd-angle-kaggle.ipynb (nesta pasta) a partir das células abaixo + gsvd_angle.py (raiz do repositório).
A biblioteca é embutida com %%writefile, então o notebook roda sozinho no Kaggle.

    python build_notebook.py
"""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": textwrap.dedent(src).strip("\n")})


# ----------------------------------------------------------------------------- 0
md("""
# GSVD + CNN: o ângulo de alinhamento θ(z) em pixels e em embeddings

Este notebook implementa o método de **[Sobrinho et al., 2026 — *GSVD for Geometry-Grounded Dataset Comparison: An Alignment Angle Is All You Need* (arXiv:2603.10283)](https://arxiv.org/abs/2603.10283)**
e o leva para onde o próprio paper aponta como trabalho futuro: **embeddings aprendidos por uma CNN**.

**A ideia em uma frase:** dados dois conjuntos A e B (ex.: imagens de "4" e de "9"), a GSVD dá um referencial
comum `A = H·C·U`, `B = H·S·V` com `CᵀC + SᵀS = I`. Para cada amostra z, compara-se o custo de explicá-la
com A versus com B:

$$\\theta(z) = \\arctan\\frac{\\|C^{+}H^{+}z\\|}{\\|S^{+}H^{+}z\\|} \\in [0, \\tfrac{\\pi}{2}]$$

θ ≈ 0° → "parece A" · θ ≈ 90° → "parece B" · θ ≈ 45° → estrutura compartilhada (ambíguo).

**Roteiro**
1. Implementação (biblioteca `gsvd_angle`, só numpy) + verificação numérica do Teorema 1
2. Reprodução do experimento do paper no MNIST (pixels): histogramas de θ, direções extremas, Fisher–Rao
3. **Extensão:** θ sobre embeddings de uma ResNet18 congelada (cães × gatos, Oxford-IIIT Pet) vs. pixels vs. regressão logística (*linear probe*)
4. Bônus: classificador multiclasse um-contra-um com θ → submissão no *Digit Recognizer*

> Contexto: a GSVD foi estudada em uma iniciação científica no IC/UFRJ da qual participei; a implementação de referência
> do grupo é em Julia. A parte de CNN/transfer learning vem do módulo 5 da pós (Coders). Créditos completos no fim.
""")

md("""
## 0. Configuração

**No Kaggle:** *Settings → Accelerator: GPU T4* (para a seção 3), *Internet: On* (download do Oxford-IIIT Pet e dos pesos
da ResNet) e *Add Input → Competitions → Digit Recognizer* (seções 2 e 4).
""")

code("""
RUN_MNIST      = True   # seção 2 — precisa do Digit Recognizer (ou internet para baixar o MNIST)
RUN_PETS       = True   # seção 3 — precisa de internet (ou de PETS_IMAGEFOLDER)
RUN_SUBMISSION = True   # seção 4 — gera submission.csv para o Digit Recognizer

# Alternativa ao Oxford-IIIT Pet: uma pasta no formato ImageFolder (cats/, dogs/), ex. o dataset do módulo 5
PETS_IMAGEFOLDER = None  # ex.: "/kaggle/input/meu-dataset-cats-dogs/train"

SEED = 42
WORK = "/kaggle/working" if __import__("os").path.isdir("/kaggle/working") else "."
""")

lib = (ROOT.parents[1] / "gsvd_angle.py").read_text()
md("## 1. A biblioteca `gsvd_angle`\n\nMesmo código de `gsvd_angle.py` do repositório (só depende de numpy).")
code("%%writefile gsvd_angle.py\n" + lib)

code("""
import glob, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import importlib, gsvd_angle
importlib.reload(gsvd_angle)
from gsvd_angle import (gsvd, alignment_angle, alignment_angle_definition, direction_angles,
                        extreme_directions, fisher_rao_histograms, posterior_A,
                        ThetaClassifier, PairwiseThetaClassifier)

rng = np.random.default_rng(SEED)
COLOR_A, COLOR_B = "#2a6fdb", "#e8833a"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
deg = np.degrees
""")

md("""
### 1.1 Verificação numérica

Construo A e B com um subespaço compartilhado S e partes exclusivas (Sa só em A, Sb só em B) e verifico:
reconstrução, `c² + s² = 1`, tamanhos dos blocos, e que **θ pela GSVD (Teorema 1) = θ pela definição**
(coeficientes de norma mínima com `Ax = By = z`) para z ∈ col(A) ∩ col(B).
""")

code("""
d, ks, ka, kb = 30, 4, 3, 5
Q, _ = np.linalg.qr(rng.normal(size=(d, ks + ka + kb)))
S, Sa, Sb = Q[:, :ks], Q[:, ks:ks+ka], Q[:, ks+ka:]
A = np.hstack([S, Sa]) @ rng.normal(size=(ks + ka, 12))   # amostras nas colunas, como no paper
B = np.hstack([S, Sb]) @ rng.normal(size=(ks + kb, 15))

g = gsvd(A, B)
Ar, Br = g.reconstruct()
print("blocos (só-A, compartilhado, só-B):", g.blocks)
print(f"‖A − HCU‖ = {np.linalg.norm(A - Ar):.1e}   ‖B − HSV‖ = {np.linalg.norm(B - Br):.1e}   "
      f"max|c²+s²−1| = {np.abs(g.c**2 + g.s**2 - 1).max():.1e}")

zs = S @ rng.normal(size=(ks, 50))
err = max(abs(alignment_angle(g, z)[0] - alignment_angle_definition(A, B, z)) for z in zs.T)
print(f"Teorema 1 — max |θ_GSVD − θ_definição| em 50 amostras: {err:.1e}")
print(f"θ(só-A) = {deg(alignment_angle(g, Sa[:, 0])[0]):.2f}°   θ(só-B) = {deg(alignment_angle(g, Sb[:, 0])[0]):.2f}°")

# mesmo caso de teste do notebook Julia da IC
g_ic = gsvd(np.array([[1, 0, 0], [0, 1, 0]], float).T, np.array([[0, 1, 0], [0, 0, 1]], float).T)
print("caso IC (plano XY × plano YZ):", g_ic.blocks, "→ ângulo da direção compartilhada:",
      f"{deg(direction_angles(g_ic)[g_ic.shared][0]):.1f}°")
""")

# ----------------------------------------------------------------------------- 2
md("""
## 2. Reprodução no MNIST (pixels)

Protocolo da Seção 5 do paper: imagens vetorizadas em ℝ⁷⁸⁴ e centradas; A = 900 imagens do dígito *a*,
B = 800 do dígito *b*; θ calculado nas imagens de teste dos dois dígitos. Como o `test.csv` do Digit Recognizer
não tem rótulos, separo 20% do `train.csv` como teste.
""")

code("""
def find_digit_recognizer():
    for path in sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True)):
        with open(path) as f:
            head = f.readline()
        if os.path.basename(path) == "train.csv" and head.startswith("label,pixel0"):
            return path, os.path.join(os.path.dirname(path), "test.csv")
    return None, None

def load_mnist():
    train_csv, test_csv = find_digit_recognizer()
    if train_csv:
        df = pd.read_csv(train_csv)
        y = df.pop("label").to_numpy()
        X = df.to_numpy(np.float32) / 255.0
        Xsub = pd.read_csv(test_csv).to_numpy(np.float32) / 255.0 if os.path.exists(test_csv) else None
        print(f"Digit Recognizer: {train_csv}  X={X.shape}")
        return X, y, Xsub
    from torchvision.datasets import MNIST           # fallback com internet
    ds = MNIST(f"{WORK}/mnist", train=True, download=True)
    X = ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    print("MNIST via torchvision", X.shape)
    return X, ds.targets.numpy(), None

if RUN_MNIST or RUN_SUBMISSION:
    X_mn, y_mn, X_sub = load_mnist()
    perm = rng.permutation(len(y_mn))
    n_test = int(0.2 * len(y_mn))
    te_idx, tr_idx = perm[:n_test], perm[n_test:]
""")

code("""
def theta_pair(X, y, a, b, p=900, q=800, rank=None, seed=SEED):
    r = np.random.default_rng(seed)
    pool_a, pool_b = np.flatnonzero(y[tr_idx] == a), np.flatnonzero(y[tr_idx] == b)
    ia = r.choice(pool_a, min(p, pool_a.size), replace=False)
    ib = r.choice(pool_b, min(q, pool_b.size), replace=False)
    clf = ThetaClassifier(rank=rank).fit(X[tr_idx][ia], X[tr_idx][ib])
    Xt, yt = X[te_idx], y[te_idx]
    return clf, clf.theta(Xt[yt == a]), clf.theta(Xt[yt == b])

def summarize(ta, tb):
    acc = 0.5 * (np.mean(ta < np.pi/4) + np.mean(tb >= np.pi/4))
    return {"acc_bal@45°": acc, "Fisher–Rao": fisher_rao_histograms(ta, tb),
            "θ̃_A (°)": deg(np.median(ta)), "θ̃_B (°)": deg(np.median(tb))}

def hist_pair(ax, ta, tb, a, b, title=""):
    bins = np.linspace(0, 90, 46)
    ax.hist(deg(ta), bins, alpha=.65, color=COLOR_A, label=f"teste '{a}'", density=True)
    ax.hist(deg(tb), bins, alpha=.65, color=COLOR_B, label=f"teste '{b}'", density=True)
    ax.axvline(45, color="k", lw=.8, ls=":")
    ax.set(xlabel="θ(z) (graus)", title=title or f"'{a}' × '{b}'", xlim=(0, 90))
    ax.legend(frameon=False, fontsize=8)

PAIRS = [(1, 5), (0, 7), (4, 9), (3, 9)]   # os mesmos pares da Fig. 2 do paper
""")

md("""
### 2.1 Histogramas de θ (Fig. 2 do paper) — posto completo vs. truncado

Com o posto completo, quase tudo satura em 0° ou 90°: em pixels, cada amostra tem um pouco de energia em
direções que só um dos conjuntos gera, e essas direções têm custo "infinito" para o outro. O paper
menciona o truncamento como opção de estabilidade numérica; aqui ele funciona como **regularização**: a GSVD
passa a usar só as `rank` direções principais de A ∪ B.
""")

code("""
if RUN_MNIST:
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharey="row")
    for k, (a, b) in enumerate(PAIRS):
        for row, rank in enumerate([None, 50]):
            _, ta, tb = theta_pair(X_mn, y_mn, a, b, rank=rank)
            rows.append({"par": f"{a}×{b}", "posto": rank or "completo", **summarize(ta, tb)})
            hist_pair(axes[row, k], ta, tb, a, b, f"'{a}' × '{b}' — posto {rank or 'completo'}")
    plt.tight_layout(); plt.show()
    display(pd.DataFrame(rows).round(3))
""")

md("### 2.2 Quanto truncar? (validação no par mais difícil, 4 × 9)")

code("""
if RUN_MNIST:
    ranks = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, None]
    res = [summarize(*theta_pair(X_mn, y_mn, 4, 9, rank=r)[1:]) for r in ranks]
    xs = [r if r else 784 for r in ranks]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(xs, [r["acc_bal@45°"] for r in res], "o-", color=COLOR_A); ax[0].set(xscale="log", xlabel="posto", ylabel="acurácia balanceada")
    ax[1].plot(xs, [r["Fisher–Rao"] for r in res], "o-", color=COLOR_B); ax[1].set(xscale="log", xlabel="posto", ylabel="Fisher–Rao (máx. π)")
    fig.suptitle("'4' × '9': efeito do truncamento (784 = posto completo)"); plt.tight_layout(); plt.show()
""")

md("""
### 2.3 Direções extremas (Fig. 3 do paper)

As colunas de H ordenadas por `arctan(s_i/c_i)` vão do "mais 4" ao "mais 9". A primeira e a última direção
compartilhada resolvem `min θ` e `max θ` (Seção 4.3); no meio fica a direção mais próxima de 45°.
""")

code("""
if RUN_MNIST:
    for rank in [None, 50]:
        clf, _, _ = theta_pair(X_mn, y_mn, 4, 9, rank=rank)
        ext = extreme_directions(clf.gsvd_)
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
        for ax, (name, e) in zip(axes[:3], ext.items()):
            ax.imshow(e["vector"].reshape(28, 28), cmap="viridis"); ax.axis("off")
            ax.set_title(f"{name.replace('_', ' ')}  (θ = {deg(e['angle']):.1f}°)")
        axes[3].plot(deg(direction_angles(clf.gsvd_)), color="k"); axes[3].set(xlabel="coluna i de H", ylabel="θ(h_i) (°)")
        fig.suptitle(f"'4' × '9' — posto {rank or 'completo'} — blocos {clf.gsvd_.blocks}"); plt.tight_layout(); plt.show()
""")

md("""
### 2.4 θ como probabilidade (Seção 6)

O paper interpreta `P(A | θ) = cos²θ`. Se a interpretação for boa, a fração real de "4" em cada faixa de θ deve
acompanhar a curva cos²θ (gráfico de calibração).
""")

code("""
if RUN_MNIST:
    clf, ta, tb = theta_pair(X_mn, y_mn, 4, 9, rank=50)
    th = np.r_[ta, tb]; is_a = np.r_[np.ones_like(ta), np.zeros_like(tb)]
    edges = np.linspace(0, np.pi/2, 13); mid = (edges[:-1] + edges[1:]) / 2
    idx = np.digitize(th, edges[1:-1])
    frac = [is_a[idx == k].mean() if np.any(idx == k) else np.nan for k in range(len(mid))]
    plt.figure(figsize=(5, 3.5))
    plt.plot(deg(mid), frac, "o-", color=COLOR_A, label="fração observada de '4'")
    plt.plot(deg(mid), posterior_A(mid), "--", color="k", label="cos²θ (paper)")
    plt.xlabel("θ (graus)"); plt.ylabel("P(A | θ)"); plt.legend(frameon=False); plt.title("Calibração de θ — '4' × '9'"); plt.show()
""")

# ----------------------------------------------------------------------------- 3
md("""
## 3. Extensão: θ sobre embeddings de CNN (cães × gatos)

O paper sugere aplicar θ a **representações aprendidas**. Aqui comparo três representações do mesmo problema:

| representação | dimensão | origem |
|---|---|---|
| pixels (32×32, cinza) | 1024 | baseline "geometria crua" |
| **embedding ResNet18 congelada** | 512 | mesma receita de transfer learning do módulo 5 (backbone ImageNet, sem treinar) |

E três classificadores: θ com limiar 45°, θ com limiar calibrado na validação e **regressão logística sobre o
embedding**, que equivale a treinar só a camada final `fc` como no `_train.py` do módulo 5.

Dados: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (37 raças, rótulo binário cão/gato),
`trainval` para A/B e `test` para avaliação.
""")

code("""
if RUN_PETS:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from torchvision.datasets import ImageFolder, OxfordIIITPet

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", DEVICE)

    tf_cnn = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   # estatísticas do ImageNet
    ])
    tf_pix = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(), transforms.ToTensor()])

    class Views(Dataset):
        \"\"\"Devolve, para cada imagem: entrada da CNN, pixels 32×32, rótulo e miniatura para as galerias.\"\"\"
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            img, label = self.base[i]
            img = img.convert("RGB")
            thumb = torch.from_numpy(np.asarray(img.resize((64, 64)), dtype=np.uint8).copy())
            return tf_cnn(img), tf_pix(img).flatten(), int(label), thumb

    def pets_splits():
        if PETS_IMAGEFOLDER:
            base = ImageFolder(PETS_IMAGEFOLDER)          # classes em ordem alfabética: cats=0, dogs=1
            idx = np.random.default_rng(SEED).permutation(len(base))
            cut = int(0.5 * len(idx))
            return (torch.utils.data.Subset(base, idx[:cut]), torch.utils.data.Subset(base, idx[cut:]))
        root = f"{WORK}/pets"
        kw = dict(root=root, target_types="binary-category", download=True)   # 0 = gato, 1 = cão
        return OxfordIIITPet(split="trainval", **kw), OxfordIIITPet(split="test", **kw)

    @torch.no_grad()
    def extract(ds, model, cache):
        if os.path.exists(cache):
            z = np.load(cache); return z["emb"], z["pix"], z["y"], z["thumb"]
        out = {"emb": [], "pix": [], "y": [], "thumb": []}
        for xc, xp, y, th in DataLoader(Views(ds), batch_size=64, num_workers=2):
            out["emb"].append(model(xc.to(DEVICE)).cpu().numpy())
            out["pix"].append(xp.numpy()); out["y"].append(y.numpy()); out["thumb"].append(th.numpy())
        out = {k: np.concatenate(v) for k, v in out.items()}
        np.savez_compressed(cache, **out)
        return out["emb"], out["pix"], out["y"], out["thumb"]

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()       # remove a camada de classificação: sai o vetor de 512 do avgpool
    backbone.eval().to(DEVICE)

    t0 = time.time()
    ds_tv, ds_te = pets_splits()
    E_tv, P_tv, y_tv, T_tv = extract(ds_tv, backbone, f"{WORK}/pets_trainval.npz")
    E_te, P_te, y_te, T_te = extract(ds_te, backbone, f"{WORK}/pets_test.npz")
    print(f"trainval {E_tv.shape} | test {E_te.shape} | gatos/cães no trainval: {np.bincount(y_tv)} | {time.time()-t0:.0f}s")
""")

md("""
### 3.1 Comparação

A = gatos, B = cães, com quantidades balanceadas. 80% do `trainval` monta A/B e 20% serve de validação
(posto e limiar). Tudo é avaliado no `test`.
""")

code("""
if RUN_PETS:
    r = np.random.default_rng(SEED)
    n_cls = np.bincount(y_tv).min()
    idx_c = r.permutation(np.flatnonzero(y_tv == 0))[:n_cls]
    idx_d = r.permutation(np.flatnonzero(y_tv == 1))[:n_cls]
    cut = int(0.8 * n_cls)
    fit_c, val_c, fit_d, val_d = idx_c[:cut], idx_c[cut:], idx_d[:cut], idx_d[cut:]
    val = np.r_[val_c, val_d]

    def bal_acc(pred, y):
        return 0.5 * (np.mean(pred[y == 0] == 0) + np.mean(pred[y == 1] == 1))

    FEATURES = {"pixels 32×32": (P_tv, P_te), "ResNet18 (512)": (E_tv, E_te)}
    RANKS = [8, 16, 32, 64, 128, 256, None]
    rows, best = [], {}
    for name, (Ftv, Fte) in FEATURES.items():
        for rank in RANKS:
            clf = ThetaClassifier(rank=rank).fit(Ftv[fit_c], Ftv[fit_d])
            val_acc = bal_acc(clf.predict(Ftv[val]), y_tv[val])     # limiar 45°
            clf.calibrate(Ftv[val], y_tv[val])
            th = clf.theta(Fte)
            rows.append({"features": name, "posto": rank or "completo", "val acc@45°": val_acc,
                         "test acc@45°": bal_acc((th > np.pi/4).astype(int), y_te),
                         "test acc calibrado": bal_acc(clf.predict(Fte), y_te),
                         "limiar (°)": deg(clf.threshold_),
                         "Fisher–Rao": fisher_rao_histograms(th[y_te == 0], th[y_te == 1])})
            if name not in best or val_acc > best[name][0]:
                best[name] = (val_acc, rank, clf, th)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    fit = np.r_[fit_c, fit_d]
    for name, (Ftv, Fte) in FEATURES.items():
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(Ftv[fit], y_tv[fit])
        rows.append({"features": name, "posto": "—", "test acc calibrado": bal_acc(lr.predict(Fte), y_te),
                     "limiar (°)": np.nan, "Fisher–Rao": np.nan, "modelo": "regressão logística (≈ fc do M5)"})

    table = pd.DataFrame(rows).fillna({"modelo": "θ (GSVD)"})
    display(table.round(3))
    for name, (v, rank, _, _) in best.items():
        print(f"melhor posto na validação — {name}: {rank or 'completo'} (val acc {v:.3f})")
""")

code("""
if RUN_PETS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (name, (_, rank, clf, th)) in zip(axes, best.items()):
        hist_pair(ax, th[y_te == 0], th[y_te == 1], "gato", "cão", f"{name} — posto {rank or 'completo'}")
        ax.axvline(deg(clf.threshold_), color="crimson", lw=1, label="limiar calibrado")
    plt.suptitle("θ(z) no conjunto de teste: pixels × embedding"); plt.tight_layout(); plt.show()
""")

md("""
### 3.2 O que θ enxerga: galerias

Com o embedding: as imagens mais "gato" (θ ≈ 0°), mais "cão" (θ ≈ 90°), as **mais ambíguas** (θ ≈ 45°)
e os erros com maior confiança. É aqui que θ ganha da regressão logística: ele mede *quanto* cada imagem é
explicada por cada conjunto, de forma interpretável.
""")

code("""
def gallery(idx_list, thumbs, thetas, labels, title, n=8):
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n, 2.2))
    for ax, i in zip(axes, idx_list[:n]):
        ax.imshow(thumbs[i]); ax.axis("off")
        ax.set_title(f"{deg(thetas[i]):.0f}° · {'cão' if labels[i] else 'gato'}", fontsize=8)
    fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.show()

if RUN_PETS:
    _, rank_e, clf_e, th_e = best["ResNet18 (512)"]
    order = np.argsort(th_e)
    gallery(order, T_te, th_e, y_te, "Mais 'gato' (θ → 0°)")
    gallery(order[::-1], T_te, th_e, y_te, "Mais 'cão' (θ → 90°)")
    gallery(np.argsort(np.abs(th_e - np.pi/4)), T_te, th_e, y_te, "Mais ambíguas (θ ≈ 45°)")
    pred = clf_e.predict(E_te)
    wrong = np.flatnonzero(pred != y_te)
    gallery(wrong[np.argsort(-np.abs(th_e[wrong] - clf_e.threshold_))], T_te, th_e, y_te, "Erros mais confiantes")
""")

md("""
### 3.3 Direções extremas no espaço do embedding

Em pixels as colunas de H viram imagens (Fig. 3). Num embedding de 512 dimensões isso não dá para desenhar,
então mostro as imagens de treino **mais alinhadas** (cosseno) com cada direção: o "mais gato", o "mais cão" e o
"compartilhado".
""")

code("""
if RUN_PETS:
    ext = extreme_directions(clf_e.gsvd_)
    Ec = E_tv - clf_e.mean_
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-12
    th_tv = clf_e.theta(E_tv)
    for name, e in ext.items():
        cos = Ec @ e["vector"]
        top = np.argsort(-np.abs(cos))
        gallery(top, T_tv, th_tv, y_tv, f"Imagens de treino mais alinhadas à direção '{name.replace('_', ' ')}' (θ(h) = {deg(e['angle']):.1f}°)")
""")

# ----------------------------------------------------------------------------- 4
md("""
## 4. Bônus: Digit Recognizer com θ (um-contra-um)

θ é binário; para 10 classes treino uma GSVD por par (45 pares) e cada par vota. Não é feito para ganhar de uma
CNN; é um classificador **sem treino por gradiente**, que só usa a geometria dos subespaços. O posto é escolhido no
holdout de 20%.
""")

code("""
if RUN_SUBMISSION:
    from sklearn.metrics import confusion_matrix
    Xtr, ytr, Xho, yho = X_mn[tr_idx], y_mn[tr_idx], X_mn[te_idx], y_mn[te_idx]
    scores = {}
    for rank in [20, 30, 50, 80]:
        t0 = time.time()
        m = PairwiseThetaClassifier(rank=rank, max_per_class=1500).fit(Xtr, ytr)
        scores[rank] = np.mean(m.predict(Xho) == yho)
        print(f"posto {rank:>3}: acurácia holdout {scores[rank]:.4f}  ({time.time()-t0:.0f}s)")
    best_rank = max(scores, key=scores.get)

    m = PairwiseThetaClassifier(rank=best_rank).fit(Xtr, ytr)
    p_ho = m.predict(Xho)
    print(f"posto {best_rank}, todas as amostras de treino → holdout {np.mean(p_ho == yho):.4f}")
    cm = confusion_matrix(yho, p_ho, normalize="true")
    plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues"); plt.colorbar(fraction=.046)
    plt.xticks(range(10)); plt.yticks(range(10)); plt.xlabel("previsto"); plt.ylabel("real"); plt.title("Matriz de confusão (holdout)"); plt.show()
""")

code("""
if RUN_SUBMISSION and X_sub is not None:
    final = PairwiseThetaClassifier(rank=best_rank).fit(X_mn, y_mn)     # treino + holdout
    sub = pd.DataFrame({"ImageId": np.arange(1, len(X_sub) + 1), "Label": final.predict(X_sub)})
    sub.to_csv(f"{WORK}/submission.csv", index=False)
    print(sub.head(), "\\nsubmission.csv salvo em", WORK)
elif RUN_SUBMISSION:
    print("test.csv do Digit Recognizer não encontrado — adicione a competição como input para gerar a submissão.")
""")

# ----------------------------------------------------------------------------- 5
md("""
## 5. Conclusões

Números da execução no Kaggle (GPU T4, seed 42); acurácia sempre balanceada.

- **Reprodução (MNIST):** o padrão do paper se repete: 0×7 é o par mais separável (Fisher–Rao 2,80 no posto completo) e
  4×9 o mais ambíguo (2,11). Os valores não são idênticos aos do paper (outro split, outras colunas sorteadas).
- **Truncamento:** com posto completo θ satura em 0° e 90°. Truncar para posto 50 leva o 4×9 de 0,857 para 0,943 de
  acurácia; a curva em função do posto é um U invertido. O posto é o hiperparâmetro que mais pesa em pixels.
- **cos²θ não é probabilidade calibrada:** no 4×9, a fração observada de "4" cai de ~0,99 para ~0,01 entre 34° e 49°,
  enquanto cos²θ prevê 0,69 → 0,43. θ ordena bem as amostras, mas a curva real é quase um degrau.
- **Pixels × embedding:** em pixels, gatos e cães ficam todos perto de 45° (acurácia 0,62, Fisher–Rao 0,61). No embedding da
  ResNet18 eles se separam: 0,971 com limiar 45°, 0,981 com limiar calibrado, Fisher–Rao 2,97 (máximo π ≈ 3,14).
- **θ × regressão logística:** 0,981 contra 0,988 da regressão logística. θ não usa gradiente e entrega, além da classe, um grau
  por amostra, as imagens ambíguas e as direções extremas. Não é mais barato: os dois ajustam em menos de um segundo.
- **Assimetria:** o limiar calibrado ficou sempre acima de 45° (46° a 54°), e os erros são quase todos gatos vistos como cães.
- **Multiclasse:** 45 GSVDs um-contra-um, posto 50: 93,15% no holdout do MNIST, sem treino por gradiente.
- **Limitações:** uma seed; a ResNet18 já viu cães e gatos no ImageNet; θ é linear e binário.

## Créditos

- Método: A. Sobrinho, E. Marques, J. Paixão, D. S. Menasché, H. Mirandola — *GSVD for Geometry-Grounded Dataset
  Comparison: An Alignment Angle Is All You Need*, [arXiv:2603.10283](https://arxiv.org/abs/2603.10283) (IC/IM — UFRJ).
- Implementação de referência da GSVD em blocos (Julia) feita pelo grupo de iniciação científica do IC/UFRJ.
- Receita de transfer learning (ResNet18 congelada + camada linear de classificação): módulo 5 da pós em IA da Coders.
- Dados: MNIST / Kaggle Digit Recognizer; Oxford-IIIT Pet (Parkhi et al., 2012).
""")

nb = {
    "cells": [{**c, "source": c["source"].splitlines(keepends=True)} for c in cells],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "gsvd-angle-kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("ok →", out)
