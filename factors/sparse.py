"""Warm-started constrained / sparse PCA.

Two complementary approaches that both start from an artificial loading
pattern (the desk's hand-drawn factor) and iterate so loadings fit the
market better while staying close to the prior:

* ``sparse_pca_warm``       — sequential power iteration with deflation;
                              unit-norm loadings; one factor at a time.
* ``soft_constrained_pca``  — joint ALS matrix factorisation with a
                              Tikhonov penalty on ``V − V0``; all k
                              factors fit simultaneously; loadings carry
                              the scale.
* ``decorr_constrained_pca``— same anchor as ``soft_constrained_pca`` plus
                              an explicit penalty on the off-diagonal of
                              ``Corr(F)``; loadings are scored via the
                              oblique projection ``F = X V (V^T V)^{-1}``
                              and the optimisation runs through PyTorch
                              autodiff.
* ``procrustes_pca_baseline`` — orthogonal Procrustes anchor (top-k PCA
                              rotated to match V0). Loadings are
                              exactly orthonormal and ``R²`` equals the
                              top-k PCA R²; serves as the "upper bound
                              on R² when loadings stay in the top-k PC
                              span" reference point on the Pareto plot.
                              Note: score correlations are NOT zero —
                              after rotation the score covariance is
                              ``R^T diag(sigma_i^2) R``, only diagonal
                              when R is identity.
* ``lambda_search``         — train/val sweep for ``soft_constrained_pca``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def sparse_pca_warm(
    wide: pd.DataFrame,
    prior: pd.Series | pd.DataFrame,
    anchor: float = 1.0,
    l1: float = 0.0,
    max_iter: int = 200,
    tol: float = 1e-7,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Sparse PCA started from an artificial loading pattern.

    Initialises loadings at ``prior`` (the desk's hand-drawn factor —
    e.g. a 2s10s steepener mask, a short-end shock, …) and iterates an
    alternating score/loading update. The loading update penalises both
    deviation from ``prior`` (Tikhonov anchor) and absolute size (L1),
    so the result is a factor that explains more market variance than
    the raw pattern but stays visually close to it.

    Parameters
    ----------
    wide       : (date × (expiry, tenor)) panel — typically already diffed.
    prior      : pd.Series indexed by ``wide.columns`` (one factor) **or**
                 pd.DataFrame whose columns match ``wide.columns`` and
                 whose rows are individual factor priors (one row → one
                 factor). The prior is normalised to unit L2 norm.
    anchor     : weight on ``||w - w_prior||²``, scaled internally by N
                 (sample size) so it's comparable to the data-fit term
                 regardless of panel length. ``0`` ≈ unconstrained PCA
                 with a warm start; ``~0.1`` lightly steers toward the
                 prior; ``~1`` is balanced; ``~10`` ≈ loadings frozen.
    l1         : optional L1 soft-threshold for extra sparsity, scaled
                 internally by N. ``0`` disables; values in ``[0, ~1]``
                 zero out cells whose data-projection magnitude is below
                 the threshold.
    max_iter   : per-factor iteration cap.
    tol        : L2 tolerance on the loading delta for early stopping.
    standardize: standardise ``wide`` cell-wise before fitting (matches
                 ``run_pca``).

    Returns
    -------
    (scores, loadings, explained) — same shapes as ``run_pca``. Loadings
    are unit-norm. ``explained`` is ``||f_i||² / ||X||_F²`` against the
    standardised panel; for multiple anchored factors the entries can
    overlap (no enforced orthogonality) so the sum may exceed each
    individual share but isn't a clean cumulative.
    """
    prior_df = prior.to_frame().T if isinstance(prior, pd.Series) else prior.copy()

    X_full = wide.dropna(axis=1)
    cols = [c for c in prior_df.columns if c in X_full.columns]
    if not cols:
        raise ValueError("prior columns do not overlap wide.columns")
    X = X_full[cols]
    prior_df = prior_df[cols]

    Xs = (StandardScaler().fit_transform(X.values) if standardize
          else X.values - X.values.mean(axis=0))

    W_prior = prior_df.values.astype(float).copy()
    norms = np.linalg.norm(W_prior, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W_prior = W_prior / norms

    k = W_prior.shape[0]
    W = np.zeros_like(W_prior)
    F = np.zeros((Xs.shape[0], k))

    N = Xs.shape[0]
    eff_anchor = anchor * N
    eff_l1 = l1 * N
    X_residual = Xs.copy()
    for i in range(k):
        w = W_prior[i].copy()
        w_prior = W_prior[i]
        for _ in range(max_iter):
            f = X_residual @ w
            denom = float(f @ f) + eff_anchor
            num = X_residual.T @ f + eff_anchor * w_prior
            if eff_l1 > 0:
                num = np.sign(num) * np.maximum(np.abs(num) - eff_l1, 0.0)
            w_new = num / denom if denom > 0 else w
            n = np.linalg.norm(w_new)
            if n > 0:
                w_new = w_new / n
            if np.linalg.norm(w_new - w) < tol:
                w = w_new
                break
            w = w_new
        f = X_residual @ w
        W[i] = w
        F[:, i] = f
        X_residual = X_residual - np.outer(f, w)

    total_var = float((Xs ** 2).sum())
    evr = (F ** 2).sum(axis=0) / total_var if total_var > 0 else np.zeros(k)

    factor_names = list(prior_df.index)
    return (
        pd.DataFrame(F, index=X.index, columns=factor_names),
        pd.DataFrame(W, index=factor_names, columns=cols),
        pd.Series(evr, index=factor_names, name="explained_variance_ratio"),
    )


def soft_constrained_pca(
    wide: pd.DataFrame,
    V0: pd.Series | pd.DataFrame,
    lam: float = 1.0,
    lambda_per_factor: np.ndarray | pd.Series | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    init: str = "prior",
    standardize: bool = True,
) -> dict:
    """Joint ALS for the soft-constrained matrix-factorisation objective::

        min_{V, F}  ||X - F V^T||_F^2  +  sum_j λ_j ||v_j - v_j^(0)||^2

    Different from ``sparse_pca_warm`` (sequential power iteration with
    deflation, unit-norm loadings) — this fits all k factors *jointly* as
    a regularised matrix factorisation. Loadings are not constrained to
    unit norm; scores carry the scale.

    Parameters
    ----------
    wide              : (T, N) DataFrame — date-indexed, MultiIndex columns.
    V0                : prior loadings. (k, N) DataFrame, one row per
                        factor, **or** (N,) Series for k=1. Codebase
                        convention — same shape as ``run_pca`` loadings.
    lam               : scalar regulariser.
    lambda_per_factor : optional length-k array; overrides ``lam`` so
                        each factor can be anchored at its own strength.
    max_iter, tol     : ALS controls — stop on relative ``||ΔV||_F / ||V||_F``.
    init              : ``'prior'`` (V = V0) or ``'pca'`` (V = top-k right
                        singular vectors of the standardised panel).
    standardize       : standardise X cell-wise before fitting.

    Returns
    -------
    dict with keys
        ``V``                    : (k, N) DataFrame, fitted loadings.
        ``F``                    : (T, k) DataFrame, fitted scores.
        ``reconstruction_error`` : ``||X - F V^T||_F^2``.
        ``loading_deviation``    : unweighted ``||V - V0||_F^2``.
        ``objective_history``    : objective per iteration (recon +
                                   λ-weighted deviation).
    """
    V0_df = V0.to_frame().T if isinstance(V0, pd.Series) else V0.copy()
    cols = [c for c in V0_df.columns if c in wide.columns]
    if not cols:
        raise ValueError("V0 columns do not overlap wide.columns")
    X_full = wide[cols].dropna(axis=0)
    V0_df = V0_df[cols]
    factor_names = list(V0_df.index)
    k = len(factor_names)

    Xs = (StandardScaler().fit_transform(X_full.values) if standardize
          else X_full.values - X_full.values.mean(axis=0))

    if lambda_per_factor is None:
        lam_vec = np.full(k, float(lam))
    else:
        lam_vec = np.asarray(lambda_per_factor, dtype=float).reshape(-1)
        if len(lam_vec) != k:
            raise ValueError(
                f"lambda_per_factor length {len(lam_vec)} != k={k}"
            )
    Lam = np.diag(lam_vec)

    V0_arr = V0_df.values.T.astype(float).copy()       # (N, k)

    if init == "prior":
        V = V0_arr.copy()
    elif init == "pca":
        _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
        V = Vt[:k].T.copy()
    else:
        raise ValueError(f"init must be 'prior' or 'pca', got {init!r}")

    def _objective(V_: np.ndarray, F_: np.ndarray) -> float:
        recon = float(np.sum((Xs - F_ @ V_.T) ** 2))
        weighted_dev = float(
            np.sum(lam_vec * np.sum((V_ - V0_arr) ** 2, axis=0))
        )
        return recon + weighted_dev

    jitter = 1e-12 * np.eye(k)
    obj_hist: list[float] = []
    V_prev = V.copy()
    F = np.zeros((Xs.shape[0], k))

    for _ in range(max_iter):
        # Step 1: solve (V^T V) F^T = V^T X^T ⇒ F = (V^T V)^{-1} V^T X^T, transposed
        VtV = V.T @ V
        F = np.linalg.solve(VtV + jitter, V.T @ Xs.T).T

        # Step 2: solve (F^T F + Lam) V^T = F^T X + Lam V0^T
        FtF = F.T @ F
        rhs = F.T @ Xs + lam_vec[:, None] * V0_arr.T   # (k, N)
        V = np.linalg.solve(FtF + Lam + jitter, rhs).T  # (N, k)

        obj_hist.append(_objective(V, F))
        if np.linalg.norm(V - V_prev) / (np.linalg.norm(V_prev) + 1e-12) < tol:
            break
        V_prev = V.copy()

    # Final F update so scores are consistent with the returned V.
    VtV = V.T @ V
    F = np.linalg.solve(VtV + jitter, V.T @ Xs.T).T

    recon_err = float(np.sum((Xs - F @ V.T) ** 2))
    load_dev = float(np.sum((V - V0_arr) ** 2))

    return {
        "V": pd.DataFrame(V.T, index=factor_names, columns=X_full.columns),
        "F": pd.DataFrame(F, index=X_full.index, columns=factor_names),
        "reconstruction_error": recon_err,
        "loading_deviation": load_dev,
        "objective_history": obj_hist,
    }


def decorr_constrained_pca(
    wide: pd.DataFrame,
    V0: pd.Series | pd.DataFrame,
    lam_anchor: float = 1.0,
    lam_decorr: float = 1.0,
    max_iter: int = 800,
    lr: float = 5e-3,
    tol: float = 1e-7,
    standardize: bool = False,
    eps: float = 1e-8,
    optimizer: str = "adam",
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Soft-constrained PCA with anchor + loading-decorrelation penalty.

    Objective (V is the only free variable; F is determined by V)::

        L(V) = ||X - F V^T||_F^2
             + lam_anchor * ||V - V0||_F^2
             + lam_decorr * ||offdiag( Corr(F) )||_F^2

        F   = X V (V^T V + eps I)^{-1}     (oblique / OLS projection)
        C_F = F^T F / T                    (k x k loading covariance)
        d   = sqrt(diag(C_F))
        Corr(F) = C_F / (d d^T)

    The oblique projection (rather than ``F = X V``) is what makes the
    pattern decomposition correct when ``V`` is non-orthogonal: each
    column of ``F`` is the OLS regression of ``X`` on that pattern with
    the others' overlap divided out. ``soft_constrained_pca`` solves a
    different problem — it lets ``F`` move freely in the joint
    factorisation — and tends to collapse all ``k`` patterns onto the
    same market direction; the correlation penalty here keeps them
    pointing in distinct directions.

    Optimised via PyTorch autodiff (Adam by default; ``optimizer='lbfgs'``
    available). ``V`` is initialised at ``V0``; no unit-norm projection
    is applied — the scale of each pattern is anchored by ``lam_anchor``.

    Parameters
    ----------
    wide        : (T, p) DataFrame of cube moves (typically already
                  diffed). Columns matching ``V0`` are used; rows with
                  NaN are dropped.
    V0          : prior loadings. (k, p) DataFrame, one row per factor,
                  or (p,) Series for k = 1. Codebase convention — same
                  shape as ``run_pca`` loadings.
    lam_anchor  : weight on ``||V - V0||_F^2``.
    lam_decorr  : weight on ``||offdiag(Corr(F))||_F^2``. The decorr
                  term is scale-invariant (correlation, not covariance)
                  so values of order ``T`` are typically needed to
                  compete with the reconstruction term.
    max_iter    : optimiser iteration cap.
    lr          : Adam learning rate. Ignored if ``optimizer='lbfgs'``.
    tol         : early-stop tolerance on ``||V_new - V_old||_F``.
    standardize : ``False`` (default) → demean each column so that
                  ``X^T X / T`` is a covariance. ``True`` → z-score
                  each column (``Sigma`` becomes correlation). The
                  decorrelation penalty is invariant to per-column
                  rescaling, but the reconstruction term is not.
    eps         : ridge added to ``V^T V`` before inversion. Increase
                  if ``cond(V^T V)`` is flagged in the output.
    optimizer   : ``'adam'`` or ``'lbfgs'``.
    device      : ``'cpu'`` or ``'cuda'``.
    verbose     : print loss components every 50 iterations.

    Returns
    -------
    dict with keys
        ``V``                    : (k, p) DataFrame, fitted loadings.
        ``F``                    : (T, k) DataFrame, oblique scores.
        ``R2``                   : ``1 - ||X(I - P_V)||^2 / ||X||^2``.
        ``reconstruction_error`` : ``||X - F V^T||_F^2``.
        ``corr_F``               : (k, k) DataFrame, ``Corr(F)``.
        ``max_offdiag_corr``     : ``max_{i!=j} |Corr(F)_{ij}|``.
        ``drift``                : per-pattern ``||v_j - v0_j||`` and
                                   ``cos(v_j, v0_j)`` and score variance.
        ``cond_VtV``             : condition number of ``V^T V``;
                                   flagged in ``warnings`` if > 1e3.
        ``warnings``             : list[str], diagnostics for the user.
        ``objective_history``    : DataFrame of total / recon / anchor
                                   / decorr per iteration.
    """
    try:
        import torch
    except ImportError as e:                                   # pragma: no cover
        raise ImportError(
            "decorr_constrained_pca requires PyTorch. "
            "Install with `pip install torch` (CPU build is fine)."
        ) from e

    V0_df = V0.to_frame().T if isinstance(V0, pd.Series) else V0.copy()
    cols = [c for c in V0_df.columns if c in wide.columns]
    if not cols:
        raise ValueError("V0 columns do not overlap wide.columns")
    X_full = wide[cols].dropna(axis=0)
    V0_df = V0_df[cols]
    factor_names = list(V0_df.index)
    k = len(factor_names)

    if standardize:
        Xs = StandardScaler().fit_transform(X_full.values)
    else:
        Xs = X_full.values - X_full.values.mean(axis=0)

    T_, p = Xs.shape
    V0_arr = V0_df.values.T.astype(float)                      # (p, k)
    total_X_sq = float((Xs ** 2).sum())

    dev = torch.device(device)
    Xt = torch.tensor(Xs, dtype=torch.float64, device=dev)
    V0t = torch.tensor(V0_arr, dtype=torch.float64, device=dev)
    V = V0t.clone().requires_grad_(True)
    eye_k = torch.eye(k, dtype=torch.float64, device=dev) * eps

    def _loss_terms():
        VtV = V.T @ V + eye_k
        VtV_inv = torch.linalg.inv(VtV)
        F_ = Xt @ V @ VtV_inv                                  # (T, k)
        recon = ((Xt - F_ @ V.T) ** 2).sum()
        anchor = ((V - V0t) ** 2).sum()
        if k > 1:
            C = F_.T @ F_ / T_
            d = torch.sqrt(torch.diagonal(C).clamp_min(1e-12))
            Corr = C / (d.unsqueeze(0) * d.unsqueeze(1))
            off = Corr - torch.diag(torch.diagonal(Corr))
            decorr = (off ** 2).sum()
        else:
            decorr = torch.tensor(0.0, dtype=torch.float64, device=dev)
        return recon, anchor, decorr

    obj_hist: list[dict] = []
    V_prev = V.detach().clone()

    if optimizer == "adam":
        opt = torch.optim.Adam([V], lr=lr)
        for it in range(max_iter):
            opt.zero_grad()
            recon, anchor, decorr = _loss_terms()
            loss = recon + lam_anchor * anchor + lam_decorr * decorr
            loss.backward()
            opt.step()
            obj_hist.append({
                "iter": it,
                "total": float(loss.detach()),
                "recon": float(recon.detach()),
                "anchor": float(anchor.detach()),
                "decorr": float(decorr.detach()),
            })
            if verbose and (it % 50 == 0 or it == max_iter - 1):
                print(f"  iter {it:4d}  total={float(loss.detach()):.4e}  "
                      f"recon={float(recon.detach()):.4e}  "
                      f"anchor={float(anchor.detach()):.4e}  "
                      f"decorr={float(decorr.detach()):.4e}")
            delta = float(torch.linalg.norm(V.detach() - V_prev))
            if it > 5 and delta < tol:
                break
            V_prev = V.detach().clone()
    elif optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            [V], lr=1.0, max_iter=max_iter, tolerance_grad=tol,
            tolerance_change=tol, line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            recon, anchor, decorr = _loss_terms()
            loss = recon + lam_anchor * anchor + lam_decorr * decorr
            loss.backward()
            obj_hist.append({
                "iter": len(obj_hist),
                "total": float(loss.detach()),
                "recon": float(recon.detach()),
                "anchor": float(anchor.detach()),
                "decorr": float(decorr.detach()),
            })
            return loss

        opt.step(closure)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}")

    with torch.no_grad():
        V_arr = V.detach().cpu().numpy()                       # (p, k)

    VtV = V_arr.T @ V_arr + eps * np.eye(k)
    VtV_inv = np.linalg.inv(VtV)
    F_arr = Xs @ V_arr @ VtV_inv                               # (T, k)

    recon_loss = float(((Xs - F_arr @ V_arr.T) ** 2).sum())
    R2 = 1.0 - recon_loss / total_X_sq if total_X_sq > 0 else 0.0

    if k > 1:
        C = F_arr.T @ F_arr / T_
        d = np.sqrt(np.maximum(np.diag(C), 1e-12))
        Corr = C / np.outer(d, d)
        off = Corr - np.diag(np.diag(Corr))
        max_off = float(np.abs(off).max())
    else:
        Corr = np.array([[1.0]])
        max_off = 0.0

    drifts = []
    for j, name in enumerate(factor_names):
        v = V_arr[:, j]
        v0 = V0_arr[:, j]
        vn = v / (np.linalg.norm(v) or 1.0)
        v0n = v0 / (np.linalg.norm(v0) or 1.0)
        drifts.append({
            "pattern": name,
            "||v - v0||": float(np.linalg.norm(v - v0)),
            "cos(v, v0)": float(vn @ v0n),
            "score_var": float(F_arr[:, j].var()),
        })

    cond_VtV = float(np.linalg.cond(VtV))
    warnings_: list[str] = []
    if cond_VtV > 1e3:
        warnings_.append(
            f"cond(V^T V) = {cond_VtV:.2e} > 1e3 — patterns are nearly "
            f"collinear; oblique projection may be unstable. Consider "
            f"increasing lam_decorr or eps."
        )

    return {
        "V": pd.DataFrame(V_arr.T, index=factor_names, columns=X_full.columns),
        "F": pd.DataFrame(F_arr, index=X_full.index, columns=factor_names),
        "R2": R2,
        "reconstruction_error": recon_loss,
        "corr_F": pd.DataFrame(Corr, index=factor_names, columns=factor_names),
        "max_offdiag_corr": max_off,
        "drift": pd.DataFrame(drifts),
        "cond_VtV": cond_VtV,
        "warnings": warnings_,
        "objective_history": pd.DataFrame(obj_hist),
    }


def procrustes_pca_baseline(
    wide: pd.DataFrame,
    V0: pd.Series | pd.DataFrame,
    standardize: bool = False,
) -> dict:
    """Orthogonal Procrustes baseline: rotate top-k PCs to match V0.

    Computes the top-k right singular vectors ``U`` of the centred panel,
    then solves ``min_{R in O(k)} ||U R - V0||_F^2`` analytically via
    SVD of ``U^T V0``. The resulting ``V = U R`` is exactly orthonormal,
    so the oblique projection collapses to the orthogonal one and
    reconstruction R² equals the top-k PCA R² regardless of how well
    the rotation matches V0.

    Note: orthonormal loadings do NOT imply uncorrelated scores. After
    the rotation, ``F^T F / T = R^T diag(sigma_i^2) R / T``, which is
    only diagonal when R is the identity. Use this as the "upper bound
    on R² when V stays in the top-k PC subspace" reference point on
    the (lam_anchor, lam_decorr) Pareto plot — it is generally NOT the
    minimum max|Corr(F)|.

    Parameters
    ----------
    wide        : (T, p) DataFrame of cube moves.
    V0          : (k, p) DataFrame or (p,) Series — only the column set
                  and dimension k are used.
    standardize : match the corresponding flag in ``decorr_constrained_pca``.

    Returns
    -------
    dict with keys ``V``, ``F``, ``R2``, ``reconstruction_error``,
    ``corr_F``, ``max_offdiag_corr``.
    """
    V0_df = V0.to_frame().T if isinstance(V0, pd.Series) else V0.copy()
    cols = [c for c in V0_df.columns if c in wide.columns]
    if not cols:
        raise ValueError("V0 columns do not overlap wide.columns")
    X_full = wide[cols].dropna(axis=0)
    V0_df = V0_df[cols]
    factor_names = list(V0_df.index)
    V0_arr = V0_df.values.T.astype(float)                      # (p, k)
    k = V0_arr.shape[1]

    if standardize:
        Xs = StandardScaler().fit_transform(X_full.values)
    else:
        Xs = X_full.values - X_full.values.mean(axis=0)
    T_ = Xs.shape[0]

    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    U = Vt[:k].T                                               # (p, k), orthonormal

    M = U.T @ V0_arr                                           # (k, k)
    A, _, Bt = np.linalg.svd(M)
    R = A @ Bt
    V_proc = U @ R                                             # (p, k), orthonormal

    F_proc = Xs @ V_proc                                       # orthogonal projection

    total_X_sq = float((Xs ** 2).sum())
    recon_loss = float(((Xs - F_proc @ V_proc.T) ** 2).sum())
    R2 = 1.0 - recon_loss / total_X_sq if total_X_sq > 0 else 0.0

    if k > 1:
        C = F_proc.T @ F_proc / T_
        d = np.sqrt(np.maximum(np.diag(C), 1e-12))
        Corr = C / np.outer(d, d)
        off = Corr - np.diag(np.diag(Corr))
        max_off = float(np.abs(off).max())
    else:
        Corr = np.array([[1.0]])
        max_off = 0.0

    return {
        "V": pd.DataFrame(V_proc.T, index=factor_names, columns=X_full.columns),
        "F": pd.DataFrame(F_proc, index=X_full.index, columns=factor_names),
        "R2": R2,
        "reconstruction_error": recon_loss,
        "corr_F": pd.DataFrame(Corr, index=factor_names, columns=factor_names),
        "max_offdiag_corr": max_off,
    }


def lambda_search(
    wide: pd.DataFrame,
    V0: pd.Series | pd.DataFrame,
    lambda_grid: list[float] | np.ndarray,
    val_frac: float = 0.2,
    init: str = "prior",
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Chronological train/val split, fit ``soft_constrained_pca`` for
    each λ in ``lambda_grid`` on the train fold, evaluate out-of-sample
    reconstruction on the val fold. Standardisation is fit on train only.

    Returns a DataFrame with columns ``lambda``,
    ``val_reconstruction_error``, ``loading_deviation`` — the Pareto
    data. **Does not auto-select λ**; pick from the curve by eye.
    """
    V0_df = V0.to_frame().T if isinstance(V0, pd.Series) else V0
    cols = [c for c in V0_df.columns if c in wide.columns]
    X_full = wide[cols].dropna(axis=0)
    V0_arr = V0_df[cols].values.T.astype(float)        # (N, k)

    n_val = max(1, int(len(X_full) * val_frac))
    X_train = X_full.iloc[:-n_val]
    X_val = X_full.iloc[-n_val:]

    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values)
    X_val_s = scaler.transform(X_val.values)

    train_df = pd.DataFrame(X_train_s, index=X_train.index,
                            columns=X_train.columns)

    rows = []
    for lam in lambda_grid:
        fit = soft_constrained_pca(
            train_df, V0_df, lam=float(lam), init=init,
            max_iter=max_iter, tol=tol, standardize=False,
        )
        V = fit["V"].values.T                          # (N, k)
        k = V.shape[1]
        VtV = V.T @ V
        F_val = np.linalg.solve(
            VtV + 1e-12 * np.eye(k), V.T @ X_val_s.T
        ).T
        val_recon = float(np.sum((X_val_s - F_val @ V.T) ** 2))
        load_dev = float(np.sum((V - V0_arr) ** 2))
        rows.append({
            "lambda": float(lam),
            "val_reconstruction_error": val_recon,
            "loading_deviation": load_dev,
        })
    return pd.DataFrame(rows)
