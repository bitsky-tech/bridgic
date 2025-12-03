import asyncio
import time
import pytest

from bridgic.core.mcp._mcp_server_connection_manager import McpServerConnectionManager


async def slow_task(duration: float, task_id: int = 0):
    """A task that sleeps for the specified duration and returns a result."""
    await asyncio.sleep(duration)
    return f"task_{task_id}_done"


@pytest.mark.asyncio
async def test_run_async_non_blocking():
    """Test that run_async allows concurrent execution of multiple tasks."""
    manager = McpServerConnectionManager.get_instance()
    
    start_time = time.time()
    
    results = await asyncio.gather(
        manager.run_async(slow_task(0.5, 1)),
        manager.run_async(slow_task(0.4, 2)),
        manager.run_async(slow_task(0.4, 3)),
    )
    
    elapsed_time = time.time() - start_time
    
    assert results[0] == "task_1_done"
    assert results[1] == "task_2_done"
    assert results[2] == "task_3_done"
    
    # If concurrent, should be ~0.5s (longest task), not ~1.2s (sum)
    assert elapsed_time < 0.6, (
        f"Expected concurrent execution (< 0.6s), but took {elapsed_time:.2f}s"
    )
    assert elapsed_time < 1.0, (
        f"Tasks appear sequential. Expected < 1.0s, got {elapsed_time:.2f}s"
    )


@pytest.mark.asyncio
async def test_run_async_timeout():
    """Test that run_async respects the timeout parameter."""
    manager = McpServerConnectionManager.get_instance()
    
    start_time = time.time()
    
    with pytest.raises(asyncio.TimeoutError):
        await manager.run_async(slow_task(2.0), timeout=0.5)
    
    elapsed_time = time.time() - start_time
    assert 0.4 < elapsed_time < 0.7, (
        f"Expected timeout around 0.5s, but took {elapsed_time:.2f}s"
    )


@pytest.mark.asyncio
async def test_run_async_multiple_managers():
    """Test that multiple manager instances can run tasks concurrently."""
    manager1 = McpServerConnectionManager(name="test-manager-1")
    manager2 = McpServerConnectionManager(name="test-manager-2")
    
    async def task(duration: float, manager_name: str):
        await asyncio.sleep(duration)
        return f"done_in_{manager_name}"
    
    start_time = time.time()
    
    results = await asyncio.gather(
        manager1.run_async(task(0.3, "manager1")),
        manager2.run_async(task(0.3, "manager2")),
    )
    
    elapsed_time = time.time() - start_time
    
    assert results[0] == "done_in_manager1"
    assert results[1] == "done_in_manager2"
    assert elapsed_time < 0.5, f"Expected concurrent execution (< 0.5s), but took {elapsed_time:.2f}s"
    
    manager1.shutdown()
    manager2.shutdown()

