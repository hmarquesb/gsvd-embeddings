"""
gsvd_angle — ângulo de alinhamento θ(z) via GSVD.

Implementa o método de:
    Sobrinho, Marques, Paixão, Menasché, Mirandola (2026).
    "GSVD for Geometry-Grounded Dataset Comparison: An Alignment Angle Is All You Need".
    arXiv:2603.10283.

Convenção (a mesma do paper): A ∈ R^{d×p} e B ∈ R^{d×q} têm AMOSTRAS NAS COLUNAS.
A GSVD é escrita na forma econômica

    A = H · diag(c) · U,     B = H · diag(s) · V,     c² + s² = 1,

com H ∈ R^{d×r} (posto coluna completo), U ∈ R^{r×p} e V ∈ R^{r×q} com linhas
ortonormais (nas direções em que c_i > 0 / s_i > 0). c é decrescente e s crescente:
as primeiras colunas de H são "mais A", as últimas "mais B".

Para uma amostra z:
    coef(z) = H⁺ z
    a(z) = ‖coef / c‖,  b(z) = ‖coef / s‖        (custos mínimos de representar z)
    θ(z) = arctan(a / b) ∈ [0, π/2]               (0 = "mais A", π/2 = "mais B", π/4 = compartilhado)

Só depende de numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = [
    "GSVDResult",
    "gsvd",
    "alignment_angle",
    "alignment_angle_definition",
    "direction_angles",
    "extreme_directions",
    "fisher_rao_histograms",
    "best_threshold",
    "ThetaClassifier",
    "PairwiseThetaClassifier",
]


# --------------------------------------------------------------------------- #
# GSVD                                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class GSVDResult:
    H: np.ndarray        # d×r   — referencial compartilhado
    c: np.ndarray        # r     — pesos de A (decrescente)
    s: np.ndarray        # r     — pesos de B (crescente)
    U: np.ndarray        # r×p
    V: np.ndarray        # r×q
    H_pinv: np.ndarray   # r×d   — H⁺, pré-calculada (H⁺ = Zᵀ Σ⁻¹ Wᵀ)
    tol: float           # limiar usado para "c_i = 0" / "s_i = 0"

    @property
    def rank(self) -> int:
        return self.H.shape[1]

    @property
    def shared(self) -> np.ndarray:
        """Índices das direções compartilhadas (c_i > 0 e s_i > 0) — o bloco central do paper."""
        return np.flatnonzero((self.c > self.tol) & (self.s > self.tol))

    @property
    def blocks(self) -> dict:
        """Tamanhos dos blocos (r, k, t) do paper: só-A, compartilhado, só-B."""
        only_a = int(np.sum(self.s <= self.tol))
        only_b = int(np.sum(self.c <= self.tol))
        return {"only_A": only_a, "shared": self.rank - only_a - only_b, "only_B": only_b}

    def reconstruct(self) -> tuple[np.ndarray, np.ndarray]:
        return self.H @ (self.c[:, None] * self.U), self.H @ (self.s[:, None] * self.V)


def _normalize_columns(M: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(M, axis=0)
    out = np.zeros_like(M)
    ok = norms > tol
    out[:, ok] = M[:, ok] / norms[ok]
    return out, norms


def gsvd(A: np.ndarray, B: np.ndarray, rtol: float | None = None,
         rank: int | None = None) -> GSVDResult:
    """
    GSVD de (A, B), com amostras nas colunas (A: d×p, B: d×q).

    Algoritmo (via decomposição CS):
      1. M = [Aᵀ; Bᵀ] = Q Σ Wᵀ (SVD truncada no posto r);   Q = [Q1; Q2], QᵀQ = I
      2. Q1 = (U1) Σ1 Zᵀ (SVD)  →  colunas de Q1·Z e Q2·Z são ortogonais,
         com normas c_i e s_i, e c_i² + s_i² = 1
      3. H = W Σ Z,  U = (Q1 Z / c)ᵀ,  V = (Q2 Z / s)ᵀ

    rtol: tolerância relativa para o posto de M (default: max(dim)·eps).
    rank: força um posto máximo (truncamento — útil com dados ruidosos).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
        raise ValueError(f"A e B precisam ter o mesmo número de linhas (d); recebi {A.shape} e {B.shape}")
    d, p = A.shape
    q = B.shape[1]

    M = np.vstack([A.T, B.T])                                    # (p+q)×d
    P, sig, Wt = np.linalg.svd(M, full_matrices=False)
    if rtol is None:
        rtol = max(M.shape) * np.finfo(np.float64).eps
    r = int(np.sum(sig > rtol * sig[0])) if sig.size and sig[0] > 0 else 0
    if rank is not None:
        r = min(r, int(rank))
    if r == 0:
        raise ValueError("A e B são nulas (posto 0).")

    Q1, Q2 = P[:p, :r], P[p:, :r]
    W, sig = Wt[:r].T, sig[:r]

    # Z: vetores singulares à direita de Q1 (r×r completo, mesmo se p < r)
    _, _, Zt = np.linalg.svd(Q1, full_matrices=True)
    Z = Zt.T

    tol = 1e-10
    U_cols, c = _normalize_columns(Q1 @ Z, tol)
    V_cols, s = _normalize_columns(Q2 @ Z, tol)

    # garante a ordem do paper (c decrescente / s crescente) mesmo com empates numéricos
    order = np.argsort(-c, kind="stable")
    c, s, Z, U_cols, V_cols = c[order], s[order], Z[:, order], U_cols[:, order], V_cols[:, order]

    H = W @ (sig[:, None] * Z)                                   # d×r
    H_pinv = Z.T @ (W.T / sig[:, None])                          # r×d
    return GSVDResult(H=H, c=c, s=s, U=U_cols.T, V=V_cols.T, H_pinv=H_pinv, tol=1e-7)


# --------------------------------------------------------------------------- #
# Ângulo de alinhamento                                                         #
# --------------------------------------------------------------------------- #
def _costs(g: GSVDResult, Z: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    coef = g.H_pinv @ Z                                          # r×n
    # piso em c/s: uma direção que A (ou B) não gera tem custo "infinito" (1/eps),
    # e não custo zero como o pseudoinverso puro daria.
    a = np.linalg.norm(coef / np.maximum(g.c, eps)[:, None], axis=0)
    b = np.linalg.norm(coef / np.maximum(g.s, eps)[:, None], axis=0)
    return a, b


def alignment_angle(g: GSVDResult, Z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """θ(z) (rad) para cada COLUNA de Z (d×n) — Teorema 1 do paper."""
    a, b = _costs(g, Z, eps)
    return np.arctan2(a, b)


def alignment_angle_definition(A: np.ndarray, B: np.ndarray, z: np.ndarray) -> float:
    """
    θ(z) pela Definição 1 (força bruta): x, y de norma mínima com Ax = By = z.
    Só faz sentido para z ∈ col(A) ∩ col(B). Usado nos testes para validar o Teorema 1.
    """
    x = np.linalg.pinv(A) @ z
    y = np.linalg.pinv(B) @ z
    return float(np.arctan2(np.linalg.norm(x), np.linalg.norm(y)))


def direction_angles(g: GSVDResult) -> np.ndarray:
    """θ(h_i) de cada coluna de H: arctan(s_i / c_i). Crescente de 0 (só-A) a π/2 (só-B)."""
    return np.arctan2(g.s, g.c)


def extreme_directions(g: GSVDResult) -> dict:
    """
    Direções representativas (Seção 4.3): z_min = h_{r+1} (primeira compartilhada, "mais A"),
    z_max = h_{r+k} (última compartilhada, "mais B") e a compartilhada mais próxima de 45°.
    """
    sh = g.shared
    if sh.size == 0:
        raise ValueError("Não há direções compartilhadas entre A e B.")
    ang = direction_angles(g)
    mid = sh[np.argmin(np.abs(ang[sh] - np.pi / 4))]
    pick = {"mais_A": sh[0], "compartilhada": mid, "mais_B": sh[-1]}
    return {k: {"index": int(i), "angle": float(ang[i]),
                "vector": g.H[:, i] / np.linalg.norm(g.H[:, i])} for k, i in pick.items()}


# --------------------------------------------------------------------------- #
# Geometria da informação (Seção 6 / Apêndice F)                                #
# --------------------------------------------------------------------------- #
def fisher_rao_histograms(theta_a: np.ndarray, theta_b: np.ndarray, bins: int = 30) -> float:
    """Distância de Fisher–Rao entre os histogramas de θ: 2·acos(Σ √(P_i Q_i)) ∈ [0, π]."""
    edges = np.linspace(0, np.pi / 2, bins + 1)
    P, _ = np.histogram(theta_a, bins=edges)
    Q, _ = np.histogram(theta_b, bins=edges)
    P = P / max(P.sum(), 1)
    Q = Q / max(Q.sum(), 1)
    bc = np.clip(np.sum(np.sqrt(P * Q)), 0.0, 1.0)
    return float(2 * np.arccos(bc))


def posterior_A(theta: np.ndarray) -> np.ndarray:
    """P(A | θ) = cos²θ (eq. 12 do paper)."""
    return np.cos(theta) ** 2


# --------------------------------------------------------------------------- #
# Classificadores                                                               #
# --------------------------------------------------------------------------- #
def best_threshold(theta: np.ndarray, y_is_b: np.ndarray) -> float:
    """Limiar que maximiza a acurácia balanceada em um conjunto de VALIDAÇÃO (y=1 → B)."""
    theta = np.asarray(theta)
    y = np.asarray(y_is_b).astype(bool)
    cand = np.unique(np.concatenate([[np.pi / 4], np.quantile(theta, np.linspace(0, 1, 201))]))
    best, best_t = -1.0, np.pi / 4
    for t in cand:
        pred = theta > t
        bal = 0.5 * (np.mean(pred[y]) if y.any() else 0) + 0.5 * (np.mean(~pred[~y]) if (~y).any() else 0)
        if bal > best:
            best, best_t = bal, t
    return float(best_t)


class ThetaClassifier:
    """
    Classificador binário a partir de θ(z) (Seção 4.4 do paper).
    API estilo sklearn, mas com amostras nas LINHAS (X: n×d), como no resto do ecossistema.
    """

    def __init__(self, center: bool = True, eps: float = 1e-8, rank: int | None = None):
        self.center, self.eps, self.rank = center, eps, rank
        self.threshold_ = np.pi / 4

    def fit(self, X_a: np.ndarray, X_b: np.ndarray) -> "ThetaClassifier":
        X_a, X_b = np.asarray(X_a, float), np.asarray(X_b, float)
        self.mean_ = np.vstack([X_a, X_b]).mean(0) if self.center else np.zeros(X_a.shape[1])
        self.gsvd_ = gsvd((X_a - self.mean_).T, (X_b - self.mean_).T, rank=self.rank)
        return self

    def theta(self, X: np.ndarray) -> np.ndarray:
        return alignment_angle(self.gsvd_, (np.asarray(X, float) - self.mean_).T, eps=self.eps)

    def calibrate(self, X_val: np.ndarray, y_val_is_b: np.ndarray) -> "ThetaClassifier":
        self.threshold_ = best_threshold(self.theta(X_val), y_val_is_b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = "mais B", 0 = "mais A"."""
        return (self.theta(X) > self.threshold_).astype(int)


class PairwiseThetaClassifier:
    """
    Multiclasse um-contra-um: uma GSVD por par de classes; cada par vota pelo lado de θ
    (θ < π/4 → classe i). Empates são decididos pela soma das margens |θ − π/4|.
    """

    def __init__(self, max_per_class: int | None = None, eps: float = 1e-8,
                 rank: int | None = None, seed: int = 0):
        self.max_per_class, self.eps, self.rank, self.seed = max_per_class, eps, rank, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PairwiseThetaClassifier":
        X, y = np.asarray(X, float), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        by_class = {}
        for k in self.classes_:
            idx = np.flatnonzero(y == k)
            if self.max_per_class and idx.size > self.max_per_class:
                idx = rng.choice(idx, self.max_per_class, replace=False)
            by_class[k] = X[idx]
        self.pairs_ = {}
        for i, j in combinations(range(len(self.classes_)), 2):
            ki, kj = self.classes_[i], self.classes_[j]
            self.pairs_[(i, j)] = ThetaClassifier(eps=self.eps, rank=self.rank).fit(by_class[ki], by_class[kj])
        return self

    def decision(self, X: np.ndarray, batch: int = 4096) -> np.ndarray:
        X = np.asarray(X, float)
        n, K = X.shape[0], len(self.classes_)
        votes = np.zeros((n, K))
        margin = np.zeros((n, K))
        for (i, j), clf in self.pairs_.items():
            th = np.concatenate([clf.theta(X[s:s + batch]) for s in range(0, n, batch)])
            m = th - np.pi / 4
            votes[:, i] += m < 0
            votes[:, j] += m >= 0
            margin[:, i] += np.maximum(-m, 0)
            margin[:, j] += np.maximum(m, 0)
        return votes + 1e-3 * margin / max(len(self.pairs_), 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision(X), axis=1)]
