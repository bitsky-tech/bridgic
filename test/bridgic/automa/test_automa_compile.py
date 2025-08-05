import pytest

from bridgic.automa import (
    GraphAutoma,
    worker,
    AutomaCompilationError,
    AutomaDeclarationError,
    WorkerSignatureError,
    AutomaRuntimeError,
)
from bridgic.automa.worker import Worker
from bridgic.utils.console import printer

def test_automa_declaration_dag_check():
    with pytest.raises(AutomaDeclarationError):
        class AutomaLayerStatic(GraphAutoma):
            @worker(is_start=True, dependencies=["worker_3"])
            def worker_0(self, *args, **kwargs) -> int:
                return 0

            @worker(dependencies=["worker_0"])
            def worker_1(self, *args, **kwargs) -> int:
                return 1

            @worker(dependencies=["worker_0"])
            def worker_2(self, *args, **kwargs) -> int:
                return 2

            @worker(dependencies=["worker_1", "worker_2"])
            async def worker_3(self, *args, **kwargs) -> int:
                return 3

def test_automa_compilation_dag_check():
    with pytest.raises(AutomaDeclarationError):
        automa_obj = GraphAutoma()

        @automa_obj.worker(dependencies=["worker_1"])
        def worker_0(atm: GraphAutoma, *args, **kwargs):
            assert atm is automa_obj

        @automa_obj.worker(dependencies=["worker_0"])
        def worker_1(atm: GraphAutoma, *args, **kwargs):
            assert atm is automa_obj

        automa_obj._compile_automa()

def test_customized_worker_signature_check():
    class IncorrectWorker(Worker):
        def process_async(self, *args, **kwargs) -> None:
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

def test_customized_worker_type_reserving():
    from layers.automa_layer_c import AutomaLayerC
    automa_obj = AutomaLayerC()
    automa_str = str(automa_obj)
    printer.print(automa_str)
    assert "CallableWorker(callable=worker_5)" in automa_str
    assert "CallableWorker(callable=<lambda>)" in automa_str
    assert "layers.automa_layer_c.PrintWorker" in automa_str
