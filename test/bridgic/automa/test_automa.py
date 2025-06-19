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
from bridgic.types.worker import LandableWorker
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
        class Worker(LandableWorker):
            async def process_async(self, *args):
                pass

        class AutomaIncorrectDecoratedWorker3(Automa):
            pass

        automa_obj = AutomaIncorrectDecoratedWorker3()
        automa_obj.register_worker(Worker(name="wrong_worker"))

    with pytest.raises(WorkerSignatureError):
        class AutomaIncorrectDecoratedWorker4(Automa):
            pass

        automa_obj = AutomaIncorrectDecoratedWorker4()
        automa_obj.add_worker(name="wrong_worker", func=lambda x: None)

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
