from rcabench_platform.v2.algorithms.spec import (
    Algorithm,
    AlgorithmAnswer,
    AlgorithmArgs,
)
from rcabench_platform.vendor.RCAEval.baro import baro
from ._common import SimpleMetricsAdapter

from functools import partial

class Baro(Algorithm):
    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> list[AlgorithmAnswer]:
        adapter = SimpleMetricsAdapter(partial(baro,dk_select_useful=True))
        return adapter(args)
