import os
import pytest

from leaf.affinity import pin_cpu, validate_worker_cpus


def test_default_does_not_change_process_affinity():
    pin_cpu(None)


@pytest.mark.parametrize("cpu", [-1, (os.cpu_count() or 1) + 1])
def test_invalid_cpu_is_rejected(cpu):
    with pytest.raises(ValueError, match="CPU index"):
        pin_cpu(cpu)


@pytest.mark.parametrize('cpus,threads', [([],1),([0,0],2),([-1],1),([True],1),([0.5],1),([0],2),('0',1),([0],True),([0],65)])
def test_worker_policy_rejects_invalid_shape_before_os_calls(cpus,threads):
    with pytest.raises(ValueError,match='one distinct'):
        validate_worker_cpus(cpus,threads)


def test_worker_policy_rejects_conflicting_single_cpu_pin():
    with pytest.raises(ValueError,match='mutually exclusive'):
        validate_worker_cpus([0],1,0)


def test_default_worker_policy_does_not_query_or_change_affinity():
    assert validate_worker_cpus(None,4) is None


def test_worker_policy_checks_actual_allowed_cpus_without_mutation():
    if os.name=='nt':
        import psutil
        process=psutil.Process(); original=process.cpu_affinity()
        cpus=original[:min(4,len(original))]
        assert validate_worker_cpus(cpus,len(cpus))==cpus
        assert process.cpu_affinity()==original
    elif hasattr(os,'sched_getaffinity'):
        original=os.sched_getaffinity(0);cpus=sorted(original)[:4]
        assert validate_worker_cpus(cpus,len(cpus))==cpus
        assert os.sched_getaffinity(0)==original
    else:
        with pytest.raises(ValueError,match='Windows and Linux'):
            validate_worker_cpus([0],1)


def test_worker_policy_rejects_unavailable_cpu():
    with pytest.raises(ValueError):
        validate_worker_cpus([10**9],1)
