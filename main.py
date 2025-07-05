#!/usr/bin/env -S uv run -s
from rcabench_platform.v2.cli.main import main
from rcabench_platform.v2.algorithms.spec import global_algorithm_registry
from baro.baro import Baro

if __name__ == "__main__":
    registry = global_algorithm_registry()
    registry["baro"] = Baro

    main(enable_builtin_algorithms=False)
