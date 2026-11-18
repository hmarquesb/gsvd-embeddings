import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsvd_angle import (  # noqa: E402
    PairwiseThetaClassifier,
    ThetaClassifier,
    alignment_angle,
    alignment_angle_definition,
    direction_angles,
    extreme_directions,
    fisher_rao_histograms,
    gsvd,
)

rng = np.random.default_rng(0)


def subspaces(d=30, k_shared=4, k_a=3, k_b=5, p=12, q=15):
    """A gera shared ⊕ só-A, B gera shared ⊕ só-B (amostras nas colunas)."""
    Q, _ = np.linalg.qr(rng.normal(size=(d, k_shared + k_a + k_b)))
    S, Sa, Sb = Q[:, :k_shared], Q[:, k_shared:k_shared + k_a], Q[:, k_shared + k_a:]
    A = np.hstack([S, Sa]) @ rng.normal(size=(k_shared + k_a, p))
    B = np.hstack([S, Sb]) @ rng.normal(size=(k_shared + k_b, q))
    return A, B, S, Sa, Sb


def test_reconstrucao_e_cs():
    A, B, *_ = subspaces()
    g = gsvd(A, B)
    Ar, Br = g.reconstruct()
    assert np.allclose(Ar, A, atol=1e-9)
    assert np.allclose(Br, B, atol=1e-9)
    assert np.allclose(g.c ** 2 + g.s ** 2, 1, atol=1e-10)
    assert np.all(np.diff(g.c) <= 1e-12)            # c decrescente
    assert np.all(np.diff(g.s) >= -1e-12)           # s crescente


def test_blocos_recuperados():
    A, B, *_ = subspaces(k_shared=4, k_a=3, k_b=5)
    g = gsvd(A, B)
    assert g.blocks == {"only_A": 3, "shared": 4, "only_B": 5}


def test_U_V_linhas_ortonormais():
    A, B, *_ = subspaces()
    g = gsvd(A, B)
    for M, w in ((g.U, g.c), (g.V, g.s)):
        on = w > 1e-7
        assert np.allclose(M[on] @ M[on].T, np.eye(on.sum()), atol=1e-8)


def test_H_pinv():
    A, B, *_ = subspaces()
    g = gsvd(A, B)
    assert np.allclose(g.H_pinv, np.linalg.pinv(g.H), atol=1e-8)


def test_teorema1_definicao_igual_gsvd():
    """θ via GSVD (Teorema 1) == θ pela Definição 1 para z ∈ col(A) ∩ col(B)."""
    A, B, S, *_ = subspaces()
    g = gsvd(A, B)
    for _ in range(20):
        z = S @ rng.normal(size=S.shape[1])
        assert alignment_angle(g, z)[0] == pytest.approx(alignment_angle_definition(A, B, z), abs=1e-7)


def test_extremos_sao_colunas_de_H():
    A, B, *_ = subspaces()
    g = gsvd(A, B)
    ext = extreme_directions(g)
    ang = direction_angles(g)
    sh = g.shared
    assert ext["mais_A"]["angle"] == pytest.approx(ang[sh].min())
    assert ext["mais_B"]["angle"] == pytest.approx(ang[sh].max())
    # nenhuma amostra do subespaço compartilhado passa dos extremos
    Z = g.H[:, sh] @ rng.normal(size=(sh.size, 200))
    th = alignment_angle(g, Z)
    assert th.min() >= ext["mais_A"]["angle"] - 1e-9
    assert th.max() <= ext["mais_B"]["angle"] + 1e-9


def test_direcoes_exclusivas():
    A, B, S, Sa, Sb = subspaces()
    g = gsvd(A, B)
    assert alignment_angle(g, Sa[:, 0])[0] < 1e-3          # só A → θ ≈ 0
    assert alignment_angle(g, Sb[:, 0])[0] > np.pi / 2 - 1e-3  # só B → θ ≈ 90°


def test_casos_do_notebook_da_IC():
    """teste1/teste2 do notebook Julia (linhas = amostras): planos que dividem o eixo y."""
    t1 = np.array([[1, 0, 0], [0, 1, 0]], float)
    t2 = np.array([[0, 1, 0], [0, 0, 1]], float)
    g = gsvd(t1.T, t2.T)
    assert g.blocks == {"only_A": 1, "shared": 1, "only_B": 1}
    assert direction_angles(g)[g.shared][0] == pytest.approx(np.pi / 4)


def test_fisher_rao():
    a = rng.uniform(0, 0.3, 500)
    b = rng.uniform(1.2, 1.5, 500)
    assert fisher_rao_histograms(a, a) == pytest.approx(0, abs=1e-9)
    assert fisher_rao_histograms(a, b) == pytest.approx(np.pi)


def test_classificadores_separam_subespacos():
    d = 40
    Qa, _ = np.linalg.qr(rng.normal(size=(d, 6)))
    Qb, _ = np.linalg.qr(rng.normal(size=(d, 6)))
    Qc, _ = np.linalg.qr(rng.normal(size=(d, 6)))
    gen = lambda Q, n: (Q @ rng.normal(size=(6, n))).T + 0.01 * rng.normal(size=(n, d))  # noqa: E731
    Xa, Xb, Xc = gen(Qa, 30), gen(Qb, 30), gen(Qc, 30)
    clf = ThetaClassifier(center=False, rank=12).fit(Xa, Xb)
    acc = np.mean(np.r_[clf.predict(gen(Qa, 50)) == 0, clf.predict(gen(Qb, 50)) == 1])
    assert acc > 0.95
    X = np.vstack([Xa, Xb, Xc])
    y = np.repeat([0, 1, 2], 30)
    mc = PairwiseThetaClassifier(rank=12).fit(X, y)
    Xt = np.vstack([gen(Qa, 40), gen(Qb, 40), gen(Qc, 40)])
    assert np.mean(mc.predict(Xt) == np.repeat([0, 1, 2], 40)) > 0.95
