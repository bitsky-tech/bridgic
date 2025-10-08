import pytest
from typing import Any
from bridgic.core.automa import GraphAutoma, using, worker
from bridgic.core.automa.worker import Worker
from bridgic.core.utils.console import printer

class PrintWorker(Worker):
    async def arun(self, *args, **kwargs) -> None:
        printer.print("  Just print something...")

@pytest.mark.asyncio
async def test_automa_single_fragment_run():
    from tests.core.automa.layers.graph_fragment_a import GraphFragmentA

    @using(GraphFragmentA)
    class TestGraphAutoma(GraphAutoma):
        @worker(is_output=True, dependencies=["loop_worker_3"])
        def output(self, result: int) -> int:
            return result

    printer.print("")
    automa_obj = TestGraphAutoma()
    result = await automa_obj.arun()
    assert result == 3

@pytest.mark.asyncio
async def test_automa_multi_fragments_run():
    from tests.core.automa.layers.graph_fragment_a import GraphFragmentA
    from tests.core.automa.layers.graph_fragment_b import GraphFragmentB

    @using(GraphFragmentA, GraphFragmentB)
    class TestGraphAutoma(GraphAutoma):
        @worker(is_output=True, dependencies=["entry_point_worker_7", "entry_point_worker_8"])
        def output(self, result_7: int, result_8: Any) -> int:
            assert result_7 == 7
            assert result_8 == None
            return 0

    automa_obj = TestGraphAutoma()
    automa_obj.set_running_options(debug=True)
    automa_obj.add_func_as_worker(
        key="entry_point_worker_7",
        func=lambda *args, **kwargs: 7,
        dependencies=[],
    )
    automa_obj.add_worker(
        key="entry_point_worker_8",
        worker=PrintWorker(),
        dependencies=[],
    )

    printer.print("")
    result = await automa_obj.arun()
    assert result == 0