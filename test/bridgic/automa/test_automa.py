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
from bridgic.utils.console import printer

def test_automa_declaration_dag_check():
    with pytest.raises(AutomaDeclarationError):
        from layers.automa_layer_wrong import AutomaLayerWrong

def test_automa_compilation_dag_check():
    with pytest.raises(AutomaDeclarationError):
        automa_obj = Automa()

        @automa_obj.worker(dependencies=["worker_1"])
        def worker_0(self):
            pass

        @automa_obj.worker(dependencies=["worker_0"])
        def worker_1(self):
            pass

        automa_obj._compile_automa()

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
