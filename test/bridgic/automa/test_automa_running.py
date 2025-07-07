import pytest

@pytest.mark.asyncio
async def test_automa_single_layer_run():
    from layers.automa_layer_a import AutomaLayerA
    automa_obj = AutomaLayerA(output_worker_name="loop_worker_3")
    result = await automa_obj.process_async(debug=False)
    assert result == 3

@pytest.mark.asyncio
async def test_automa_multi_layer_run():
    from layers.automa_layer_c import AutomaLayerC
    automa_obj = AutomaLayerC(output_worker_name="entry_point_worker_7")
    result = await automa_obj.process_async(debug=False)
    assert result == 7

@pytest.mark.asyncio
async def test_automa_classic_10_workers_run():
    from layers.automa_layer_c import AutomaLayerC, PrintWorker, EndWorker
    automa_obj = AutomaLayerC(output_worker_name="customized_end_worker_10")
    automa_obj.add_worker(PrintWorker(name="worker_9"), dependencies=["entry_point_worker_7", "entry_point_worker_8"])
    automa_obj.add_worker(EndWorker(name="customized_end_worker_10"), dependencies=["worker_9"])
    result = await automa_obj.process_async(debug=False)
    assert result == "happy_ending"

@pytest.mark.asyncio
async def test_automa_nesting_run():
    from layers.automa_layer_d import AutomaLayerD
    from layers.automa_layer_f import AutomaLayerF

    automa_obj = AutomaLayerD(
        name="easy_start_automa",
        output_worker_name="continue_automa",
    )
    automa_obj.add_worker(
        AutomaLayerF(
            name="continue_automa",
            output_worker_name="transfer_datetime_worker",
        )
    )
    result = await automa_obj.process_async(debug=True)
    assert len(result) == len("2025-06-20 12:30:00")