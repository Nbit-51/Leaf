import os
import pytest

from leaf.affinity import pin_cpu


def test_default_does_not_change_process_affinity():
    pin_cpu(None)


@pytest.mark.parametrize("cpu", [-1, (os.cpu_count() or 1) + 1])
def test_invalid_cpu_is_rejected(cpu):
    with pytest.raises(ValueError, match="CPU index"):
        pin_cpu(cpu)
