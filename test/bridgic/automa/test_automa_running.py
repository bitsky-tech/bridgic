import pytest

@pytest.mark.asyncio
async def test_automa_single_layer_run():
    from layers.automa_layer_a import AutomaLayerA
    automa_obj = AutomaLayerA(output_worker_key="loop_worker_3")
    result = await automa_obj.process_async()
    assert result == 3

@pytest.mark.asyncio
async def test_automa_multi_layer_run():
    from layers.automa_layer_c import AutomaLayerC
    automa_obj = AutomaLayerC(output_worker_key="entry_point_worker_7")
    result = await automa_obj.process_async()
    assert result == 7

from layers.automa_layer_d import AutomaLayerD
from layers.automa_layer_f import AutomaLayerF

@pytest.fixture
def ready_automa_obj():
    automa_obj = AutomaLayerD(name="wrapped_automa", output_worker_key="continue_automa")
    automa_obj.set_running_options(debug=True)
    automa_obj.add_worker(
        key="continue_automa",
        worker_obj=AutomaLayerF(
            output_worker_key="transfer_datetime_worker",
        ),
        dependencies=[],
    )
    yield automa_obj

def test_automa_naming(ready_automa_obj: AutomaLayerD):
    assert ready_automa_obj.name == "wrapped_automa"

@pytest.mark.asyncio
async def test_automa_nesting_run(ready_automa_obj: AutomaLayerD):
    result = await ready_automa_obj.process_async()
    assert len(result) == len("2025-06-20 12:30:00")

@pytest.mark.asyncio
async def test_automa_classic_10_workers_run(ready_automa_obj: AutomaLayerD):
    from layers.automa_layer_c import AutomaLayerC, EndWorker
    automa_obj = AutomaLayerC(output_worker_key="customized_end_worker_10")
    automa_obj.set_running_options(debug=True)
    automa_obj.add_worker("nested_automa_9", ready_automa_obj, dependencies=["entry_point_worker_7", "entry_point_worker_8"])
    automa_obj.add_worker("customized_end_worker_10", EndWorker(), dependencies=["nested_automa_9"])
    result = await automa_obj.process_async()
    assert result == "happy_ending"
