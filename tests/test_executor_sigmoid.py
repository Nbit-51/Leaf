import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "graph_opt"))

from executor import _sigmoid


def test_sigmoid():
    x = np.array([-1.0, 0.0, 1.0])

    result = _sigmoid(x)
    expected = 1 / (1 + np.exp(-x))

    np.testing.assert_allclose(result, expected)
