from typing import Any, Dict
from bridgic.core.automa import ArgsMappingRule, AutomaRuntimeError, GraphAutoma, worker
import pytest

#### Test case: worker_state runtime_context must contain worker_key
class MissingWorkerKeyStateAutoma(GraphAutoma):
    @worker(is_start=True)
    async def start(self):
        loop_index, set_loop_index = self.worker_state(1, runtime_context = {})
        set_loop_index(loop_index + 1)
        return loop_index
    
    @worker(dependencies=["start"])
    async def end(self, loop_index: int):
        return loop_index + 5

@pytest.fixture
def missing_worker_key_state_automa():
    graph = MissingWorkerKeyStateAutoma(output_worker_key="end")
    return graph

@pytest.mark.asyncio
async def test_worker_state_requires_worker_key(missing_worker_key_state_automa: MissingWorkerKeyStateAutoma):
    with pytest.raises(AutomaRuntimeError, match=r"the worker_key is not found in the runtime_context:"):
        await missing_worker_key_state_automa.arun()

#### Test case: worker_state cannot be called multiple times within a single worker_state

class DuplicateWorkerStateCallAutoma(GraphAutoma):
    @worker(is_start=True)
    async def start(self):
        loop_index, set_loop_index = self.worker_state(1, runtime_context = {
            "worker_key": "start"
        })
        loop_index, set_loop_index = self.worker_state(1, runtime_context = {
            "worker_key": "start"
        })
        set_loop_index(loop_index + 1)
        return loop_index
    
    @worker(dependencies=["start"])
    async def end(self, loop_index: int):
        return loop_index + 5

@pytest.fixture
def duplicate_worker_state_call_automa():
    graph = DuplicateWorkerStateCallAutoma(output_worker_key="end")
    return graph

@pytest.mark.asyncio
async def test_worker_state_called_more_than_once_raises(duplicate_worker_state_call_automa: DuplicateWorkerStateCallAutoma):
    with pytest.raises(AutomaRuntimeError, match=r".* has been run more than once by the same worker_state call"):
        await duplicate_worker_state_call_automa.arun()

#### Test case: worker state can be saved and reused across runs

class ArithmeticAutomaWithState(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        start_state, set_start_state = self.worker_state(x,runtime_context = {
            "worker_key": "start"
        })
        # start_state first run is the default value 2; second run reads from worker state 3*2; third run reads from worker state 3 * (3*2)
        new_start_state = 3 * start_state
        # set the loop_index
        set_start_state(new_start_state)
        return new_start_state

    @worker(dependencies=["start"])
    async def end(self, start_state: int):
        end_state, set_end_state = self.worker_state(start_state, runtime_context = {
            "worker_key": "end"
        })
        # end_state first run is the default value 3 * 2; second run reads from worker state 3*2 + 5; third run reads from worker state 3*2 + 5 + 5
        new_end_state = end_state + 5
        # set the end_state
        set_end_state(new_end_state)
        return new_end_state # the arun result value is the new_end_state

@pytest.fixture
def arithmetic_automa_with_state():
    graph = ArithmeticAutomaWithState(output_worker_key="end")
    return graph

@pytest.mark.asyncio
async def test_automa_rerun_persists_worker_state(arithmetic_automa_with_state: ArithmeticAutomaWithState):
    # First run.
    result = await arithmetic_automa_with_state.arun(x=2)
    assert result == 2*3+5
    # Second run (with worker state). x=5 does not take effect, because worker_state is bound to the Automa instance. After rerun, the value is the last set_start_state, so start_state is 3*2
    result = await arithmetic_automa_with_state.arun(x=5)
    assert result == (3 * 2 + 5) + 5
    assert result != 3*5+5
    # Third run (with worker state). x=10 does not take effect, because worker_state is bound to the Automa instance. After rerun, the value is the last set_start_state, so start_state is 3 * (3*2)
    result = await arithmetic_automa_with_state.arun(x=10)
    assert result == (3*2 + 5 + 5) + 5
    assert result != 3 * 10 + 5


#### Test case: rerun a nested Automa via ferry_to; nested counters should persist across reruns



class TodoItem():
    def __init__(self, text, completed=False, learn_count=1):
        self.text = text
        self.completed = completed
        self.learn_count = learn_count
    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "completed": self.completed,
            "learn_count": self.learn_count
        }
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.text = state_dict["text"]
        self.completed = state_dict["completed"]
        self.learn_count = state_dict["learn_count"]
    def __str__(self):
        status = "✅" if self.completed else "⭕"
        return f"{status} {self.text} {self.learn_count}"
    
    def __repr__(self):
        return f"TodoItem('{self.text}', {self.completed}, {self.learn_count})"
    
    def __eq__(self, other):
        return (isinstance(other, TodoItem) and 
                self.text == other.text and 
                self.completed == other.completed and
                self.learn_count == other.learn_count)

class TopAutoma(GraphAutoma):
    # The start worker is a nested Automa which will be added by add_worker()

    @worker(dependencies=["start"])
    async def end(self, my_list: list[str]):
        if len(my_list) < 5:
            self.ferry_to("start")
        else:
            return my_list

class NestedAutoma(GraphAutoma):

    @worker(is_start=True)
    async def start(self):
        loop_index, set_loop_index = self.worker_state(1, runtime_context = {
            "worker_key": "start"
        })
        set_loop_index(loop_index + 1)
        return loop_index

    
    @worker(dependencies=["start"])
    async def test_worker_state_str(self, loop_index):
        hi, set_hi = self.worker_state("hi", runtime_context = {
            "worker_key": "test_worker_state_str"
        })
        hi = hi + str(loop_index)
        set_hi(hi)
        if loop_index == 1:
            assert hi == "hi1"
        elif loop_index == 2:
            assert hi == "hi12"
        elif loop_index == 3:
            assert hi == "hi123"
        elif loop_index == 4:
            assert hi == "hi1234"
        elif loop_index == 5:
            assert hi == "hi12345"
        return loop_index

    @worker(dependencies=["test_worker_state_str"])
    async def test_worker_state_obj(self, loop_index):
        count_obj, set_count = self.worker_state({
            "count": 1
        }, runtime_context = {
            "worker_key": "test_worker_state_obj"
        })
        set_count({
            "count": count_obj.get("count") + 1
        })
        assert count_obj.get("count") == loop_index
        return loop_index

    
    @worker(dependencies=["test_worker_state_obj"])
    async def test_worker_state_class(self,loop_index):

        todo, set_todo = self.worker_state(TodoItem("Learn Bridgic"), runtime_context = {
            "worker_key": "test_worker_state_class"
        })
        assert todo.text == "Learn Bridgic"
        assert type(todo).__name__ == "TodoItem"

        # Update todo state
        completed_todo = TodoItem("Learn Bridgic", True, 2)
        if loop_index == 1:
            assert todo.text == "Learn Bridgic"
            assert type(todo).__name__ == "TodoItem"
            assert not todo.completed
            assert todo.learn_count == 1
        elif loop_index == 2:
            assert todo.text == "Learn Bridgic"
            assert type(todo).__name__ == "TodoItem"
            assert todo.completed
            assert todo.learn_count == 2
        set_todo(completed_todo)

    @worker(dependencies=["test_worker_state_class"], args_mapping_rule=ArgsMappingRule.SUPPRESSED)
    async def test_worker_state_int(self):
        count, set_count = self.worker_state(0, runtime_context = {
            "worker_key": "test_worker_state_int"
        })
        set_count(count + 1)
        return count
    
    @worker(dependencies=["test_worker_state_int"])
    async def end(self, count: int):
        return ['Learn Bridgic'] * count

@pytest.fixture
def nested_automa():
    graph = NestedAutoma(output_worker_key="end")
    return graph

@pytest.fixture
def top_automa_with_nested(nested_automa):
    graph = TopAutoma(output_worker_key="end")
    graph.add_worker("start", nested_automa, is_start=True)
    return graph

@pytest.mark.asyncio
async def test_nested_automa_rerun_preserves_state(top_automa_with_nested):
    result = await top_automa_with_nested.arun()
    assert result == ['Learn Bridgic'] * 5


#### Test case: worker state serialization and deserialization

class SerializationStateAutoma(GraphAutoma):
    @worker(is_start=True)
    async def start(self):
        count, set_count = self.worker_state(1, runtime_context = {
            "worker_key": "start"
        })
        set_count(count + 1)
        return count
    
    @worker(dependencies=["start"])
    async def end(self, loop_index: int):
        return loop_index + 5

@pytest.fixture
def serialization_state_automa():
    graph = SerializationStateAutoma(output_worker_key="end")
    return graph

@pytest.mark.asyncio
async def test_automa_serialization_preserves_worker_state(serialization_state_automa: SerializationStateAutoma):
    result = await serialization_state_automa.arun()
    assert result == 1 + 5
    automa_dict = serialization_state_automa.dump_to_dict()
    serialization_state_automa.load_from_dict(automa_dict)
    result = await serialization_state_automa.arun()
    assert result == 2 + 5


#####: Test case: worker state serialization/deserialization with a complex object (no nested Automa)

class ComplexObjectStateAutoma(GraphAutoma):
    @worker(is_start=True)
    async def start(self):
        loop_index, set_loop_index = self.worker_state(1, runtime_context = {
            "worker_key": "start"
        })
        set_loop_index(loop_index + 1)
        return loop_index
    
    @worker(dependencies=["start"])
    async def test_item(self, loop_index: int):
        item, set_item = self.worker_state(TodoItem("Learn Bridgic"), runtime_context = {
            "worker_key": "test_item"
        })
        new_item = TodoItem("Learn Bridgic" + str(loop_index), loop_index % 2 == 0, loop_index + 1)
        set_item(new_item)
        if loop_index == 1:
            assert item.text == "Learn Bridgic"
            assert item.completed == False
            assert item.learn_count == 1
        elif loop_index == 2:
            assert item.text == "Learn Bridgic1"
            assert item.completed == False
            assert item.learn_count == 2
        elif loop_index == 3:
            assert item.text == "Learn Bridgic2"
            assert item.completed == True
            assert item.learn_count == 3
        elif loop_index == 4:
            assert item.text == "Learn Bridgic3"
            assert item.completed == False
            assert item.learn_count == 4
        elif loop_index == 5:
            assert item.text == "Learn Bridgic4"
            assert item.completed == True
            assert item.learn_count == 5
        return item
    
    @worker(dependencies=["test_item"])
    async def end(self, item: TodoItem):
        if item.learn_count < 5:
            self.ferry_to("start")
        else:
            return ["Learn Bridgic"] * item.learn_count

@pytest.fixture
def complex_object_state_automa():
    graph = ComplexObjectStateAutoma(output_worker_key="end")
    return graph


@pytest.mark.asyncio
async def test_complex_object_state_automa_serialization(complex_object_state_automa: ComplexObjectStateAutoma):
    result = await complex_object_state_automa.arun()
    assert result == ["Learn Bridgic"] * 5
    automa_dict = complex_object_state_automa.dump_to_dict()
    complex_object_state_automa.load_from_dict(automa_dict)
    assert isinstance(complex_object_state_automa, ComplexObjectStateAutoma)
    # TODO: add post_arun() to Automa or auto_clear() to worker_state to clear the worker_state after arun()
    result = await complex_object_state_automa.arun()
    assert result == ["Learn Bridgic"] * 6 # because the serialization is usefulful, so the learn_count is 6 > 5
    assert isinstance(complex_object_state_automa, ComplexObjectStateAutoma)


#####: Test case: worker state serialization/deserialization with a complex object (with nested Automa)

class ComplexObjectStateNestedAutoma(ComplexObjectStateAutoma):
        @worker(dependencies=["test_item"])
        def nested_end(self, item: TodoItem):
                return ["Learn Bridgic"] * item.learn_count

@pytest.fixture
def complex_object_state_nested_automa():
    graph = ComplexObjectStateNestedAutoma(output_worker_key="nested_end")
    return graph

@pytest.fixture
def top_automa_with_complex_nested(complex_object_state_nested_automa: ComplexObjectStateNestedAutoma):
    graph = TopAutoma(output_worker_key="end")
    graph.add_worker("start", complex_object_state_nested_automa, is_start=True)
    return graph

@pytest.mark.asyncio
async def test_complex_object_nested_automa_serialization(top_automa_with_complex_nested: TopAutoma):
    result = await top_automa_with_complex_nested.arun()
    assert result == ["Learn Bridgic"] * 5
    automa_dict = top_automa_with_complex_nested.dump_to_dict()
    top_automa_with_complex_nested.load_from_dict(automa_dict)
    assert isinstance(top_automa_with_complex_nested, TopAutoma)
    # TODO: add post_arun() to Automa or auto_clear() to worker_state to clear the worker_state after arun()
    result = await top_automa_with_complex_nested.arun()
    assert result == ["Learn Bridgic"] * 6 # because the serialization is usefulful, so the learn_count is 6 > 5
    assert isinstance(top_automa_with_complex_nested, TopAutoma)