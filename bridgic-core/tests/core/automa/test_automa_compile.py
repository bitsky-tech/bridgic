import pytest

from bridgic.core.automa import (
    GraphAutoma,
    worker,
    AutomaCompilationError,
    WorkerSignatureError,
)
from bridgic.core.automa.worker import Worker
from bridgic.core.utils._console import printer

# TODO: This file need to be refactored later...
def test_automa_declaration_dag_check():
    with pytest.raises(AutomaCompilationError):
        class AutomaLayerStatic(GraphAutoma):
            @worker(is_start=True, dependencies=["worker_3"])
            async def worker_0(self, *args, **kwargs) -> int:
                return 0

            @worker(dependencies=["worker_0"])
            async def worker_1(self, *args, **kwargs) -> int:
                return 1

            @worker(dependencies=["worker_0"])
            async def worker_2(self, *args, **kwargs) -> int:
                return 2

            @worker(dependencies=["worker_1", "worker_2"])
            async def worker_3(self, *args, **kwargs) -> int:
                return 3

def test_automa_compilation_dag_check():
    with pytest.raises(AutomaCompilationError):
        automa_obj = GraphAutoma()

        @automa_obj.worker(dependencies=["worker_1"])
        async def worker_0(atm: GraphAutoma, *args, **kwargs):
            assert atm is automa_obj

        @automa_obj.worker(dependencies=["worker_0"])
        async def worker_1(atm: GraphAutoma, *args, **kwargs):
            assert atm is automa_obj

        automa_obj._compile_graph_and_detect_risks()

def test_customized_worker_signature_check():
    class IncorrectWorker(Worker):
        def arun(self, *args, **kwargs) -> None:
            pass

    with pytest.raises(WorkerSignatureError):
        class AutomaBackground(GraphAutoma):
            pass

        automa_obj = AutomaBackground()
        automa_obj.add_worker("incorrect_worker", IncorrectWorker())

    with pytest.raises(WorkerSignatureError, match="Unexpected arguments"):
        class AutomaBackground(GraphAutoma):
            @worker(is_start=True, wrong_parameter=True)
            async def start(self, *args, **kwargs) -> None:
                pass
