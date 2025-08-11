import concurrent.futures
import functools
import multiprocessing
import sys
import time
from pathlib import Path

from rcabench.openapi import (
    AlgorithmsApi,
    DatasetsApi,
    DtoGranularityResultEnhancedRequest,
    DtoGranularityResultItem,
)
from rcabench_platform.v2.algorithms.spec import (
    AlgorithmArgs,
    global_algorithm_registry,
)
from rcabench_platform.v2.clients.rcabench_ import RCABenchClient
from rcabench_platform.v2.logging import logger, timeit
from rcabench_platform.v2.sources.convert import convert_datapack
from rcabench_platform.v2.sources.rcabench import RcabenchDatapackLoader

from baro.baro import Baro


# Top-level helper so it’s picklable
def _run_callable(fn):
    return fn()


@timeit()
def get_dataset(id: int):
    with RCABenchClient() as client:
        api = DatasetsApi(client)
        resp = api.api_v2_datasets_id_get(id=id, include_injections=True)
        assert resp.data is not None

    assert resp.data.injections is not None
    return resp.data.injections


@timeit()
def run_job(algorithm, algorithm_id: int, injection_id: int, injection_name: str):
    """
    Plain, top-level worker function (no decorators, no Typer annotations).
    Safe to use from multiprocessing pools.
    """
    input_path = Path("data") / "rcabench_dataset" / injection_name
    converted_input_path = input_path / "converted"

    convert_datapack(
        loader=RcabenchDatapackLoader(src_folder=input_path, datapack=injection_name),
        dst_folder=converted_input_path,
        skip_finished=True,
    )

    a = algorithm()

    answers = a(
        AlgorithmArgs(
            dataset="rcabench",
            datapack=injection_name,
            input_folder=converted_input_path,
            output_folder=Path("/tmp/baro_output") / injection_name,
        )
    )

    result_rows = [
        {"level": ans.level, "result": ans.name, "rank": ans.rank, "confidence": 0}
        for ans in answers
    ]

    with RCABenchClient() as client:
        algo_api = AlgorithmsApi(client)

        resp = algo_api.api_v2_algorithms_algorithm_id_results_post(
            algorithm_id=algorithm_id,
            execution_id=None,
            request=DtoGranularityResultEnhancedRequest(
                results=[
                    DtoGranularityResultItem(
                        level=row["level"],
                        result=row["result"],
                        rank=row["rank"],
                        confidence=row["confidence"],
                    )
                    for row in result_rows
                ],
                datapack_id=injection_id,
            ),
        )
        logger.info(
            f"Submit detector result: response code: {resp.code}, message: {resp.message}"
        )


def run_batch(
    algorithm,
    algorithm_id: int,
    datasets: list[int],
    *,
    use_cpus: int | None = None,
    ignore_exceptions: bool = True,
):
    logger.debug(f"algorithms=`{algorithm}`")

    for dataset in datasets:
        datapacks_injections = get_dataset(dataset)

        alg = algorithm()
        alg_cpu_count = alg.needs_cpu_count()

        if alg_cpu_count is None:
            parallel = 0
        else:
            assert alg_cpu_count > 0
            usable_cpu_count = use_cpus or max(multiprocessing.cpu_count() - 4, 0)
            parallel = usable_cpu_count // alg_cpu_count

        del alg

        tasks = []
        for datapack in datapacks_injections:
            tasks.append(
                functools.partial(
                    run_job,
                    algorithm,
                    algorithm_id,
                    datapack.id,
                    datapack.injection_name,
                )
            )

        t0 = time.time()

        if parallel and parallel > 1:
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=parallel, mp_context=ctx
            ) as ex:
                list(ex.map(_run_callable, tasks))
        else:
            for task in tasks:
                task()

        t1 = time.time()

        total_walltime = t1 - t0
        avg_walltime = total_walltime / len(tasks)

        logger.debug(f"Total   walltime: {total_walltime:.3f} seconds")
        logger.debug(f"Average walltime: {avg_walltime:.3f} seconds")

        logger.debug(f"Finished running algorithm `{algorithm}` on dataset `{dataset}`")

        sys.stdout.flush()


if __name__ == "__main__":
    registry = global_algorithm_registry()
    registry["baro"] = Baro
    get_dataset(2)
    run_batch(
        algorithm=Baro,
        algorithm_id=49,
        datasets=[2, 3, 4, 5],
    )
