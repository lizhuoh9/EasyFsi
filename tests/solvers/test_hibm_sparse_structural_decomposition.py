"""The sparse structural solve preserves the dense normalized SVD contract."""

from unittest import mock

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from simulation_core.coupling.hibm_mpm import marker_mac_constraint as constraints


def test_sparse_structure_matches_dense_svd_and_minimum_norm():
    rng = np.random.default_rng(82)
    # Exact dependent rows and a weak independent direction. The rank cutoff
    # acts on singular values, not their squares or inverse-mass weights.
    base = rng.normal(size=(6, 11))
    base[5] *= 1.0e-7
    matrix = np.vstack((base, base[:4], np.zeros((2, 11))))
    sparse = csr_matrix(matrix)
    norm = np.linalg.norm(matrix, axis=0)
    normalized = matrix / norm
    row_basis, lift = constraints._sparse_structural_decomposition(
        sparse, column_norm=norm, rcond=1.0e-12
    )
    assert row_basis.shape[1] == 6
    np.testing.assert_allclose(normalized @ lift, row_basis, atol=5.0e-9, rtol=5.0e-9)
    rhs = rng.normal(size=matrix.shape[0])
    expected = np.linalg.lstsq(normalized, rhs, rcond=1.0e-12)[0]
    actual = lift @ (row_basis.T @ rhs)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-7, atol=1.0e-7)


def test_sparse_structure_checks_budget_before_growing_rank():
    matrix = csr_matrix(np.eye(10))
    with mock.patch.object(
        constraints, "HIBM_MARKER_PRESSURE_NULLSPACE_RESOURCE_MAX_BYTES", 64
    ):
        with pytest.raises(RuntimeError, match="memory budget"):
            constraints._sparse_structural_decomposition(
                matrix, column_norm=np.ones(10), rcond=1.0e-12
            )
