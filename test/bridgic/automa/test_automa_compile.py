import pytest

from bridgic.automa import (
    Automa,
    worker,
    AutomaCompilationError,
    AutomaDeclarationError,
    WorkerSignatureError,
    AutomaRuntimeError,
)
from bridgic.automa.worker import Worker

def test_automa_declaration_dag_check():
    with pytest.raises(AutomaDeclarationError):
        class AutomaLayerStatic(Automa):
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
        automa_obj = Automa()

        @automa_obj.worker(dependencies=["worker_1"])
        def worker_0(atm: Automa, *args, **kwargs):
            assert atm is automa_obj

        @automa_obj.worker(dependencies=["worker_0"])
        def worker_1(atm: Automa, *args, **kwargs):
            assert atm is automa_obj

        automa_obj._compile_automa()

def test_customized_worker_signature_check():
    class IncorrectWorker(Worker):
        def process_async(self, *args, **kwargs) -> None:
            pass
    
    with pytest.raises(WorkerSignatureError):
        class AutomaIncorrectDecoratedWorker5(Automa):
            pass

        automa_obj = AutomaIncorrectDecoratedWorker5()
        automa_obj.add_worker(IncorrectWorker(name="wrong_worker"))
