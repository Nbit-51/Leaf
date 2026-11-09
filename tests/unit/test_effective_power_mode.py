"""Mode transitions must fail timing even when AC and the saver bit look good."""
from types import SimpleNamespace
import ctypes
import json

import pytest

from leaf import power


@pytest.fixture
def windows_modes(monkeypatch):
    config = dict(initial=3, register_result=0, unregister_result=0, handle=123, calls=[])
    def register(version, callback, context, handle):
        config['calls'].append(('register', version))
        config['callback'] = callback
        handle._obj.value = config['handle']
        if config['initial'] is not None:
            callback(config['initial'], None)
        return config['register_result']
    def unregister(handle):
        config['calls'].append(('unregister', handle.value))
        if 'at_unregister' in config:
            config['callback'](config['at_unregister'], None)
        return config['unregister_result']
    dll = SimpleNamespace(PowerRegisterForEffectivePowerModeNotifications=register,
                          PowerUnregisterFromEffectivePowerModeNotifications=unregister)
    monkeypatch.setattr(power, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **k: dll, raising=False)
    monkeypatch.setattr(ctypes, 'WINFUNCTYPE', ctypes.CFUNCTYPE, raising=False)
    monkeypatch.setattr(power, '_POWER_MODE_INITIAL_TIMEOUT_SECONDS', .001)
    monkeypatch.setattr(power, '_POWER_MODE_ORPHANED_CALLBACKS', [])
    config['dll'] = dll
    return config


@pytest.mark.parametrize('mode', [2, 3, 4, 5, 6])
def test_constant_performance_modes_are_audited_and_cleaned_up(windows_modes, mode):
    windows_modes['initial'] = mode
    record = {}
    with power.benchmark_power_mode_guard(record):
        assert not record['passed']
    assert record['passed'] and record['initial_mode'] == record['final_mode']
    assert not record['system_power_plan_changed']
    assert windows_modes['calls'] == [('register', 2), ('unregister', 123)]
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize('mode', [0, 1, -1, 99])
def test_saving_or_unknown_mode_prevents_measurement(windows_modes, mode):
    windows_modes['initial'] = mode
    record = {}
    with pytest.raises(RuntimeError, match='saving/unknown|unknown mode'):
        with power.benchmark_power_mode_guard(record):
            pytest.fail('No benchmark may start')
    assert not record['passed']
    assert windows_modes['calls'][-1] == ('unregister', 123)


@pytest.mark.parametrize('mode', [0, 1, 2, 4, 99])
def test_transition_and_return_to_original_cannot_qualify(windows_modes, mode):
    record = {}
    with pytest.raises(RuntimeError, match='changed|saving/unknown|unknown mode'):
        with power.benchmark_power_mode_guard(record):
            windows_modes['callback'](mode, None)
            windows_modes['callback'](3, None)
    assert record['initial_mode'] == record['final_mode'] == 'high-performance'
    assert not record['passed'] and len(record['notifications']) == 3


def test_duplicate_same_mode_callbacks_do_not_fail(windows_modes):
    record = {}
    with power.benchmark_power_mode_guard(record):
        windows_modes['callback'](3, None)
    assert record['passed']


def test_last_callback_during_unregistration_is_validated(windows_modes):
    windows_modes['at_unregister'] = 4
    record = {}
    with pytest.raises(RuntimeError, match='changed'):
        with power.benchmark_power_mode_guard(record):
            pass
    assert not record['passed'] and record['final_mode'] == 'max-performance'


def test_body_failure_is_preserved_and_cleanup_runs(windows_modes):
    record = {}
    with pytest.raises(ValueError, match='original benchmark failure'):
        with power.benchmark_power_mode_guard(record):
            windows_modes['callback'](1, None)
            raise ValueError('original benchmark failure')
    assert not record['passed']
    assert windows_modes['calls'][-1] == ('unregister', 123)


def test_required_mode_is_checked_before_benchmark(windows_modes):
    record = {}
    with pytest.raises(RuntimeError, match='requires effective power mode max-performance'):
        with power.benchmark_power_mode_guard(record, required_mode='max-performance'):
            pytest.fail('Wrong mode cannot start timing')
    assert not record['passed']


def test_matching_required_mode_passes(windows_modes):
    record = {}
    with power.benchmark_power_mode_guard(record, required_mode='high-performance'):
        pass
    assert record['passed']


@pytest.mark.parametrize('mode,name', [(0,'battery-saver'), (1,'better-battery')])
def test_explicit_matched_efficiency_comparison_preserves_user_policy(windows_modes, mode, name):
    windows_modes['initial'] = mode
    record = {}
    with power.benchmark_power_mode_guard(record, required_mode=name): pass
    assert record['passed'] and record['saving_mode_explicitly_selected']
    assert record['required_mode'] == name and not record['system_power_plan_changed']


def test_explicit_saving_mode_does_not_allow_boost_then_return(windows_modes):
    windows_modes['initial'] = 1
    record = {}
    with pytest.raises(RuntimeError, match='changed'):
        with power.benchmark_power_mode_guard(record, required_mode='better-battery'):
            windows_modes['callback'](3, None)
            windows_modes['callback'](1, None)
    assert not record['passed']


def test_no_initial_callback_fails_and_unregisters(windows_modes):
    windows_modes['initial'] = None
    with pytest.raises(RuntimeError, match='no initial'):
        with power.benchmark_power_mode_guard({}):
            pytest.fail('Unknown current mode cannot start timing')
    assert windows_modes['calls'][-1] == ('unregister', 123)


def test_registration_failure_does_not_claim_ownership(windows_modes):
    windows_modes['register_result'] = -5
    record = {}
    with pytest.raises(OSError, match='register effective'):
        with power.benchmark_power_mode_guard(record):
            pass
    assert windows_modes['calls'] == [('register', 2)] and not record['passed']


@pytest.mark.parametrize('fault', ['null-handle', 'unregister-failed'])
def test_callback_address_is_retained_when_cleanup_cannot_be_confirmed(windows_modes, fault):
    if fault == 'null-handle': windows_modes['handle'] = None
    else: windows_modes['unregister_result'] = -5
    record = {}
    with pytest.raises((OSError, RuntimeError)):
        with power.benchmark_power_mode_guard(record):
            pass
    assert not record['passed'] and len(power._POWER_MODE_ORPHANED_CALLBACKS) == 1


def test_missing_api_fails_before_benchmark(windows_modes):
    del windows_modes['dll'].PowerRegisterForEffectivePowerModeNotifications
    with pytest.raises(RuntimeError, match='unavailable'):
        with power.benchmark_power_mode_guard({}): pass
    assert windows_modes['calls'] == []


def test_existing_record_cannot_be_overwritten(windows_modes):
    record = {'previous': 'retain'}
    with pytest.raises(ValueError, match='fresh empty'):
        with power.benchmark_power_mode_guard(record): pass
    assert record == {'previous': 'retain'} and windows_modes['calls'] == []


def test_non_windows_is_not_a_successful_audit(monkeypatch):
    monkeypatch.setattr(power, 'os', SimpleNamespace(name='posix'))
    with pytest.raises(RuntimeError, match='requires Windows'):
        with power.benchmark_power_mode_guard({}): pass
