import numpy as np
import pytest

from tools.verify_embeddinggemma_reference import mean_pool, project_normalize


def test_pooling_excludes_padding_but_includes_every_unmasked_token():
    hidden = np.array([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]], dtype=np.float32)
    np.testing.assert_array_equal(
        mean_pool(hidden, np.array([[1, 1, 0]])), [[2.0, 4.0]]
    )


@pytest.mark.parametrize(
    "mask", [np.array([[0, 0]]), np.array([[1, 2]]), np.array([[1]])]
)
def test_invalid_or_empty_pooling_mask_is_rejected(mask):
    with pytest.raises(ValueError):
        mean_pool(np.ones((1, 2, 3)), mask)


def test_projection_applies_both_heads_before_unit_normalization():
    pooled = np.array([[2.0, 1.0]], dtype=np.float32)
    first = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    second = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(
        project_normalize(pooled, first, second), [[2 / np.sqrt(29), 5 / np.sqrt(29)]]
    )


def test_zero_projection_cannot_claim_a_normalized_embedding():
    with pytest.raises(ValueError, match="Zero"):
        project_normalize(np.zeros((1, 2)), np.eye(2), np.eye(2))


def test_incorrect_projection_orientation_is_rejected():
    with pytest.raises(ValueError, match="shapes"):
        project_normalize(np.ones((1, 2)), np.ones((2, 3)), np.ones((2, 3)))
