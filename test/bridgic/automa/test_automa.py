import pytest
import asyncio

from bridgic.automa import (
    Automa,
    worker,
    AutomaCompilationError,
    AutomaDeclarationError,
    WorkerSignatureError,
    AutomaRuntimeError,
)
from bridgic.types.worker import Worker
from bridgic.utils.console import printer

def test_automa_declaration_dag_check():
    with pytest.raises(AutomaDeclarationError):
        from layers.automa_layer_wrong import AutomaLayerWrong

def test_automa_compilation_dag_check():
    with pytest.raises(AutomaDeclarationError):
        automa_obj = Automa()

        @automa_obj.worker(dependencies=["worker_1"])
        def worker_0(self, *args, **kwargs):
            pass

        @automa_obj.worker(dependencies=["worker_0"])
        def worker_1(self, *args, **kwargs):
            pass

        automa_obj._compile_automa()

def test_decorated_worker_signature_check():
    with pytest.raises(WorkerSignatureError):
        class AutomaIncorrectDecoratedWorker1(Automa):
            @worker(dependencies=["worker_1"])
            def worker_0(self, **kwargs):
                pass

            @worker(dependencies=["worker_0"])
            def worker_1(self, *args, **kwargs):
                pass

    with pytest.raises(WorkerSignatureError):
        class AutomaIncorrectDecoratedWorker2(Automa):
            pass

        automa_obj = AutomaIncorrectDecoratedWorker2()

        @automa_obj.worker(dependencies=["worker_1"])
        def func(self, **kwargs):
            pass

def test_later_added_worker_signature_check():
    with pytest.raises(WorkerSignatureError):
        class IncorrectWorker(Worker):
            async def process_async(self, *args):
                pass

        class AutomaIncorrectDecoratedWorker3(Automa):
            pass

        automa_obj = AutomaIncorrectDecoratedWorker3()
        automa_obj.add_worker(IncorrectWorker(name="wrong_worker"))

    with pytest.raises(WorkerSignatureError):
        class AutomaIncorrectDecoratedWorker4(Automa):
            pass

        automa_obj = AutomaIncorrectDecoratedWorker4()
        automa_obj.add_func_as_worker(name="wrong_worker", func=lambda x: None)

def test_customized_worker_signature_check():
    class IncorrectWorker(Worker):
        def process_async(self, *args, **kwargs) -> None:
            pass
    
    with pytest.raises(WorkerSignatureError):
        class AutomaIncorrectDecoratedWorker5(Automa):
            pass

        automa_obj = AutomaIncorrectDecoratedWorker5()
        automa_obj.add_worker(IncorrectWorker(name="wrong_worker"))

@pytest.mark.asyncio
async def test_automa_single_layer_run():
    from layers.automa_layer_a import AutomaLayerA
    automa_obj = AutomaLayerA()
    await automa_obj.process_async(debug=False)

@pytest.mark.asyncio
async def test_automa_multi_layer_run():
    from layers.automa_layer_c import AutomaLayerC
    automa_obj = AutomaLayerC()
    await automa_obj.process_async(debug=False)

@pytest.mark.asyncio
async def test_automa_classic_10_workers_run():
    from layers.automa_layer_c import AutomaLayerC, PrintWorker
    automa_obj = AutomaLayerC()
    automa_obj.add_worker(PrintWorker(name="worker_9"), dependencies=["entry_point_worker_7", "entry_point_worker_8"])
    automa_obj.add_worker(PrintWorker(name="customized_end_worker_10"), dependencies=["worker_9"])
    await automa_obj.process_async(debug=True)
