"""Factor-construction extensions to vanilla ``pca.py``.

All work below operates on **daily diffs** of the wide panels — we
care about cube moves, not levels. Every helper is unit-agnostic and
simply takes a wide DataFrame.

Organised into family-specific sub-modules; everything is re-exported
here so ``from factors import X`` keeps working:

* ``factors.rotation``    — ``varimax``, ``rotate_scores``
* ``factors.blocks``      — block PCA (partition the cube into a grid,
                            PCA each block, stitch back)
* ``factors.sparse``      — ``sparse_pca_warm``, ``soft_constrained_pca``,
                            ``lambda_search`` (anchored / warm-started PCA)
* ``factors.regression``  — ``regress`` (generic OLS), ``project_onto_patterns``
                            (cross-sectional projection on a fixed basis)
* ``factors.cca``         — ``cross_surface_cca``, ``lagged_corr``
                            (joint structure between score panels)
* ``factors.separable``   — Kronecker / functional PCA family:
                            ``marginal_kronecker_cov``,
                            ``kronecker_cov_mle``,
                            ``kronecker_separability_residual``,
                            ``roughness_penalty_1d`` / ``_2d``,
                            ``functional_pca``,
                            ``marginal_eigen_patterns``.
* ``factors.metrics``     — ``variance_retained``, ``loading_sparsity``,
                            ``rolling_stability``, ``replication_residual``,
                            ``metrics_table``
"""

from .rotation import varimax, rotate_scores
from .blocks import (
    DEFAULT_EXPIRY_BLOCKS,
    DEFAULT_TENOR_BLOCKS,
    make_blocks,
    block_pca,
    reconstruct_block,
    stack_block_scores,
    block_summary,
)
from .sparse import (
    sparse_pca_warm,
    soft_constrained_pca,
    decorr_constrained_pca,
    procrustes_pca_baseline,
    lambda_search,
)
from .regression import regress, project_onto_patterns
from .cca import cross_surface_cca, lagged_corr
from .separable import (
    marginal_kronecker_cov,
    kronecker_cov_mle,
    kronecker_separability_residual,
    roughness_penalty_1d,
    roughness_penalty_2d,
    functional_pca,
    marginal_eigen_patterns,
)
from .metrics import (
    variance_retained,
    loading_sparsity,
    rolling_stability,
    replication_residual,
    metrics_table,
)

__all__ = [
    # rotation
    "varimax", "rotate_scores",
    # blocks
    "DEFAULT_EXPIRY_BLOCKS", "DEFAULT_TENOR_BLOCKS",
    "make_blocks", "block_pca", "reconstruct_block",
    "stack_block_scores", "block_summary",
    # sparse
    "sparse_pca_warm", "soft_constrained_pca",
    "decorr_constrained_pca", "procrustes_pca_baseline",
    "lambda_search",
    # regression
    "regress", "project_onto_patterns",
    # cca
    "cross_surface_cca", "lagged_corr",
    # separable
    "marginal_kronecker_cov", "kronecker_cov_mle",
    "kronecker_separability_residual",
    "roughness_penalty_1d", "roughness_penalty_2d",
    "functional_pca", "marginal_eigen_patterns",
    # metrics
    "variance_retained", "loading_sparsity", "rolling_stability",
    "replication_residual", "metrics_table",
]
