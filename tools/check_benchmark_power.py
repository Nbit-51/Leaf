"""Read-only Windows benchmark power-mode preflight; never changes settings."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaf.power import benchmark_power_mode_guard, require_benchmark_ac, WINDOWS_POWER_MODE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--required-mode', choices=WINDOWS_POWER_MODE_NAMES)
    args = parser.parse_args()
    result = dict(passed=False, power_mode={})
    try:
        result['ac'] = require_benchmark_ac()
        with benchmark_power_mode_guard(result['power_mode'], required_mode=args.required_mode):
            pass
        result['passed'] = True
    except (OSError, RuntimeError) as error:
        result['error'] = str(error)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
