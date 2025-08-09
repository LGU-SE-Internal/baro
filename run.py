from pathlib import Path
from typing import Annotated

import typer
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

app = typer.Typer()


@app.command()
@timeit()
def get_dataset(id: int):
    with RCABenchClient() as client:
        api = DatasetsApi(client)
        resp = api.api_v2_datasets_id_get(id=id, include_injections=True)
        assert resp.data is not None

    assert resp.data.injections is not None
    return resp.data.injections


@app.command()
@timeit()
def run(
    algorithm: Annotated[str, typer.Option("-a", "--algorithm", envvar="ALGORITHM")],
    injection_id: Annotated[int, typer.Option(help="Injection ID")],
    injection_name: Annotated[str, typer.Option(help="Injection name")],
):
    assert algorithm in global_algorithm_registry(), f"Unknown algorithm: {algorithm}"
    input_path = Path("data") / "rcabench_dataset" / injection_name

    converted_input_path = input_path / "converted"

    convert_datapack(
        loader=RcabenchDatapackLoader(src_folder=input_path, datapack=injection_name),
        dst_folder=converted_input_path,
        skip_finished=True,
    )

    a = global_algorithm_registry()[algorithm]()

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
            algorithm_id=3,
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


if __name__ == "__main__":
    registry = global_algorithm_registry()
    registry["baro"] = Baro
    run("baro", 264, "ts1-ts-ui-dashboard-request-delay-6m8m29")
