"""
Integration tests for nested Automa human interaction scenarios.

These tests verify human-in-the-loop interaction mechanism with nested
Automa instances and multiple simultaneous interactions.
"""
import pytest
import asyncio

from bridgic.core.automa import GraphAutoma
from bridgic.core.automa import worker
from bridgic.core.automa.interaction import Event
from bridgic.core.automa.interaction import InteractionFeedback, InteractionException
from bridgic.core.automa import Snapshot


################################################################################
# Shared base classes for nested Automa tests
################################################################################

class TopGraph(GraphAutoma):
    """Top level graph used as container for nested automa tests."""
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 1

    # The 'middle' worker is an Automa, which will be added by add_worker() method.

    @worker(dependencies=["middle"], is_output=True)
    async def end(self, x: int):
        return x + 2


class SecondLayerGraph(GraphAutoma):
    """Second layer graph without human interaction, used as base for nested tests."""
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 10

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        return x + 20

    # The 'func_2' worker is an Automa, which will be added by add_worker() method.

    @worker(dependencies=["func_1", "func_2"], is_output=True)
    async def end(self, x1: int, x2: int):
        return x1 + x2


# Utility function to deserialize a TopGraph from a bytes file and a version file.
def deserialize_top_graph(bytes_file, version_file):
    serialized_bytes = bytes_file.read_bytes()
    serialization_version = version_file.read_text()
    snapshot = Snapshot(
        serialized_bytes=serialized_bytes, 
        serialization_version=serialization_version
    )
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_graph = TopGraph.load_from_snapshot(snapshot)
    assert type(deserialized_graph) is TopGraph
    return deserialized_graph


################################################################################
# Test Case 1: Multiple simultaneous human interactions in parallel branches
################################################################################

class ThirdLayerGraph_1_ParallelBranches(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 100

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        event = Event(
            event_type="if_add",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 200 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 200
        return x

    @worker(dependencies=["start"])
    async def func_2(self, x: int):
        return x + 300

    @worker(dependencies=["start"])
    async def func_3(self, x: int):
        event = Event(
            event_type="if_add",  # event_type do not need to be unique.
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 400 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 400
        return x

    @worker(dependencies=["func_1", "func_2", "func_3"], is_output=True)
    async def end(self, x1: int, x2: int, x3: int):
        return x1 + x2 + x3


@pytest.fixture
def graph_1_third_layer():
    return ThirdLayerGraph_1_ParallelBranches()


@pytest.fixture
def graph_1_second_layer(graph_1_third_layer):
    graph = SecondLayerGraph()
    graph.add_worker("func_2", graph_1_third_layer, dependencies=["start"])
    return graph


@pytest.fixture
def graph_1(graph_1_second_layer):
    graph = TopGraph()
    graph.add_worker("middle", graph_1_second_layer, dependencies=["start"])
    return graph


@pytest.mark.asyncio
async def test_graph_1_serialized(graph_1: TopGraph, request, db_base_path):
    """Test nested Automa with multiple simultaneous human interactions."""
    try:
        result = await graph_1.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 2
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add"
            assert "Current value is 116" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids_1", interaction_ids)
        bytes_file = db_base_path / "graph_1.bytes"
        version_file = db_base_path / "graph_1.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)


@pytest.fixture
def graph_1_deserialized(db_base_path):
    bytes_file = db_base_path / "graph_1.bytes"
    version_file = db_base_path / "graph_1.version"
    return deserialize_top_graph(bytes_file, version_file)


@pytest.fixture
def interaction_feedbacks_for_graph_1(request):
    interaction_ids = request.config.cache.get("interaction_ids_1", None)
    feedback_no = InteractionFeedback(interaction_id=interaction_ids[0], data="no")
    feedback_yes = InteractionFeedback(interaction_id=interaction_ids[1], data="yes")
    return [feedback_yes, feedback_no]


@pytest.mark.asyncio
async def test_graph_1_deserialized(interaction_feedbacks_for_graph_1, graph_1_deserialized):
    """Test nested Automa with multiple simultaneous human interactions after deserialization."""
    result = await graph_1_deserialized.arun(feedback_data=interaction_feedbacks_for_graph_1)
    assert result == 1286 - 200


################################################################################
# Test Case 2: One branch is human interaction, another branch is nested Automa
################################################################################

class SecondLayerGraph_2_Parallel_Interaction(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 10

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        event = Event(
            event_type="if_add",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 20 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 20
        return x

    @worker(dependencies=["func_1", "func_2"], is_output=True)
    async def end(self, x1: int, x2: int):
        return x1 + x2


class ThirdLayerGraph_2(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 100

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        await asyncio.sleep(0.0)  # A chance to switch to concurrent tasks
        return x + 200

    @worker(dependencies=["start"])
    async def func_2(self, x: int):
        return x + 300

    @worker(dependencies=["start"])
    async def func_3(self, x: int):
        return x + 400

    @worker(dependencies=["func_1", "func_2", "func_3"], is_output=True)
    async def end(self, x1: int, x2: int, x3: int):
        return x1 + x2 + x3


@pytest.fixture
def graph_2_third_layer():
    return ThirdLayerGraph_2()


@pytest.fixture
def graph_2_second_layer(graph_2_third_layer):
    graph = SecondLayerGraph_2_Parallel_Interaction()
    graph.add_worker("func_2", graph_2_third_layer, dependencies=["start"])
    return graph


@pytest.fixture
def graph_2(graph_2_second_layer):
    graph = TopGraph()
    graph.add_worker("middle", graph_2_second_layer, dependencies=["start"])
    return graph


@pytest.mark.asyncio
async def test_graph_2_serialized(graph_2: TopGraph, request, db_base_path):
    """Test parallel human interaction with nested Automa."""
    try:
        result = await graph_2.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 1
        assert e.interactions[0].event.event_type == "if_add"
        assert "Current value is 16" in e.interactions[0].event.data["prompt_to_user"]
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids_2", interaction_ids)
        bytes_file = db_base_path / "graph_2.bytes"
        version_file = db_base_path / "graph_2.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)


@pytest.fixture
def graph_2_deserialized(db_base_path):
    bytes_file = db_base_path / "graph_2.bytes"
    version_file = db_base_path / "graph_2.version"
    return deserialize_top_graph(bytes_file, version_file)


@pytest.fixture
def feedback_no_for_graph_2(request):
    interaction_ids = request.config.cache.get("interaction_ids_2", None)
    return InteractionFeedback(interaction_id=interaction_ids[0], data="no")


@pytest.mark.asyncio
async def test_graph_2_deserialized(feedback_no_for_graph_2, graph_2_deserialized):
    """Test resuming after human interaction with nested Automa."""
    result = await graph_2_deserialized.arun(feedback_data=feedback_no_for_graph_2)
    assert result == 1286 - 20


################################################################################
# Test Case 3: Multiple human interactions in a single worker
################################################################################

class SecondLayerGraph_3_Multiple_Interactions(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 10

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        # First human interaction
        event = Event(
            event_type="if_add_1",
            data={"prompt_to_user": f"Current value is {x}, do you want to add another 20 to it (yes/no) ?"}
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 20

        # Second human interaction
        event = Event(
            event_type="if_add_2",
            data={"prompt_to_user": f"Current value is {x}, do you want to add another 30 to it (yes/no) ?"}
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 30

        # Third human interaction
        event = Event(
            event_type="if_add_3",
            data={"prompt_to_user": f"Current value is {x}, do you want to add another 40 to it (yes/no) ?"}
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 40

        return x

    @worker(dependencies=["func_1", "func_2"], is_output=True)
    async def end(self, x1: int, x2: int):
        return x1 + x2


@pytest.fixture
def graph_3_second_layer(graph_2_third_layer):
    graph = SecondLayerGraph_3_Multiple_Interactions()
    graph.add_worker("func_2", graph_2_third_layer, dependencies=["start"])
    return graph


@pytest.fixture
def graph_3(graph_3_second_layer):
    graph = TopGraph()
    graph.add_worker("middle", graph_3_second_layer, dependencies=["start"])
    return graph


@pytest.mark.asyncio
async def test_graph_3_first_interaction(graph_3: TopGraph, request, db_base_path):
    """Test first human interaction in sequential interactions."""
    try:
        await graph_3.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 1
        assert e.interactions[0].event.event_type == "if_add_1"
        assert "Current value is 16" in e.interactions[0].event.data["prompt_to_user"]
        request.config.cache.set("interaction_ids_3", [e.interactions[0].interaction_id])
        (db_base_path / "graph_3.bytes").write_bytes(e.snapshot.serialized_bytes)
        (db_base_path / "graph_3.version").write_text(e.snapshot.serialization_version)


@pytest.fixture
def graph_3_deserialized_first(db_base_path):
    return deserialize_top_graph(db_base_path / "graph_3.bytes", db_base_path / "graph_3.version")


@pytest.mark.asyncio
async def test_graph_3_second_interaction(graph_3_deserialized_first, request, db_base_path):
    """Test second human interaction in sequential interactions."""
    interaction_ids = request.config.cache.get("interaction_ids_3", None)
    feedback = InteractionFeedback(interaction_id=interaction_ids[0], data="yes")
    try:
        await graph_3_deserialized_first.arun(feedback_data=feedback)
    except InteractionException as e:
        assert len(e.interactions) == 1
        assert e.interactions[0].event.event_type == "if_add_2"
        assert "Current value is 36" in e.interactions[0].event.data["prompt_to_user"]
        request.config.cache.set("interaction_ids_3", [e.interactions[0].interaction_id])
        (db_base_path / "graph_3.bytes").write_bytes(e.snapshot.serialized_bytes)
        (db_base_path / "graph_3.version").write_text(e.snapshot.serialization_version)


@pytest.fixture
def graph_3_deserialized_second(db_base_path):
    return deserialize_top_graph(db_base_path / "graph_3.bytes", db_base_path / "graph_3.version")


@pytest.mark.asyncio
async def test_graph_3_third_interaction(graph_3_deserialized_second, request, db_base_path):
    """Test third human interaction in sequential interactions."""
    interaction_ids = request.config.cache.get("interaction_ids_3", None)
    feedback = InteractionFeedback(interaction_id=interaction_ids[0], data="no")
    try:
        await graph_3_deserialized_second.arun(feedback_data=feedback)
    except InteractionException as e:
        assert len(e.interactions) == 1
        assert e.interactions[0].event.event_type == "if_add_3"
        assert "Current value is 36" in e.interactions[0].event.data["prompt_to_user"]
        request.config.cache.set("interaction_ids_3", [e.interactions[0].interaction_id])
        (db_base_path / "graph_3.bytes").write_bytes(e.snapshot.serialized_bytes)
        (db_base_path / "graph_3.version").write_text(e.snapshot.serialization_version)


@pytest.fixture
def graph_3_deserialized_third(db_base_path):
    return deserialize_top_graph(db_base_path / "graph_3.bytes", db_base_path / "graph_3.version")


@pytest.mark.asyncio
async def test_graph_3_final(graph_3_deserialized_third, request):
    """Test final completion after all sequential interactions."""
    interaction_ids = request.config.cache.get("interaction_ids_3", None)
    feedback = InteractionFeedback(interaction_id=interaction_ids[0], data="yes")
    result = await graph_3_deserialized_third.arun(feedback_data=feedback)
    assert result == 1286 + 40  # yes, no, yes -> 16 + 20 + 40 = 76 in func_1


################################################################################
# Test Case 4: Simultaneous human interactions in different layers
################################################################################

class SecondLayerGraph_4(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 10

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        event = Event(
            event_type="if_add_in_second_layer",
            data={"prompt_to_user": f"Current value is {x}, do you want to add another 20 to it (yes/no) ?"}
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 20
        return x

    @worker(dependencies=["func_1", "func_2"], is_output=True)
    async def end(self, x1: int, x2: int):
        return x1 + x2


class ThirdLayerGraph_4(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 100

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        return x + 200

    @worker(dependencies=["start"])
    async def func_2(self, x: int):
        event = Event(
            event_type="if_add_in_third_layer",
            data={"prompt_to_user": f"Current value is {x}, do you want to add another 300 to it (yes/no) ?"}
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 300
        return x

    @worker(dependencies=["start"])
    async def func_3(self, x: int):
        return x + 400

    @worker(dependencies=["func_1", "func_2", "func_3"], is_output=True)
    async def end(self, x1: int, x2: int, x3: int):
        return x1 + x2 + x3


@pytest.fixture
def graph_4_third_layer():
    return ThirdLayerGraph_4()


@pytest.fixture
def graph_4_second_layer(graph_4_third_layer):
    graph = SecondLayerGraph_4()
    graph.add_worker("func_2", graph_4_third_layer, dependencies=["start"])
    return graph


@pytest.fixture
def graph_4(graph_4_second_layer):
    graph = TopGraph()
    graph.add_worker("middle", graph_4_second_layer, dependencies=["start"])
    return graph


@pytest.mark.asyncio
async def test_graph_4_serialized(graph_4: TopGraph, request, db_base_path):
    """Test simultaneous human interactions in different layers."""
    try:
        await graph_4.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 2
        assert e.interactions[0].event.event_type == "if_add_in_second_layer"
        assert "Current value is 16" in e.interactions[0].event.data["prompt_to_user"]
        assert e.interactions[1].event.event_type == "if_add_in_third_layer"
        assert "Current value is 116" in e.interactions[1].event.data["prompt_to_user"]
        interaction_ids = [i.interaction_id for i in e.interactions]
        request.config.cache.set("interaction_ids_4", interaction_ids)
        (db_base_path / "graph_4.bytes").write_bytes(e.snapshot.serialized_bytes)
        (db_base_path / "graph_4.version").write_text(e.snapshot.serialization_version)


@pytest.fixture
def graph_4_deserialized(db_base_path):
    return deserialize_top_graph(db_base_path / "graph_4.bytes", db_base_path / "graph_4.version")


@pytest.fixture
def interaction_feedbacks_for_graph_4(request):
    interaction_ids = request.config.cache.get("interaction_ids_4", None)
    return [
        InteractionFeedback(interaction_id=interaction_ids[0], data="yes"),
        InteractionFeedback(interaction_id=interaction_ids[1], data="yes"),
    ]


@pytest.mark.asyncio
async def test_graph_4_deserialized(interaction_feedbacks_for_graph_4, graph_4_deserialized):
    """Test resuming after simultaneous interactions in different layers."""
    result = await graph_4_deserialized.arun(feedback_data=interaction_feedbacks_for_graph_4)
    assert result == 1286


################################################################################
# Test Case 5: Sequential human interactions in different layers
################################################################################

class TopGraph_5(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 1

    @worker(dependencies=["middle"], is_output=True)
    async def end(self, x: int):
        event = Event(
            event_type="if_add_in_top_layer",
            data={"prompt_to_user": f"Current value is {x}, do you want to add another 2 to it (yes/no) ?"}
        )
        # This interact_with_human() will occur after the third layer's human interaction.
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 2
        return x


class ThirdLayerGraph_5(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 100

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        return x + 200

    @worker(dependencies=["start"])
    async def func_2(self, x: int):
        event = Event(
            event_type="if_add_in_third_layer",
            data={"prompt_to_user": f"Current value is {x}, do you want to add another 300 to it (yes/no) ?"}
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 300
        return x

    @worker(dependencies=["start"])
    async def func_3(self, x: int):
        return x + 400

    @worker(dependencies=["func_1", "func_2", "func_3"], is_output=True)
    async def end(self, x1: int, x2: int, x3: int):
        return x1 + x2 + x3


def deserialize_top_graph_5(bytes_file, version_file):
    serialized_bytes = bytes_file.read_bytes()
    serialization_version = version_file.read_text()
    snapshot = Snapshot(serialized_bytes=serialized_bytes, serialization_version=serialization_version)
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_graph = TopGraph_5.load_from_snapshot(snapshot)
    assert type(deserialized_graph) is TopGraph_5
    return deserialized_graph


@pytest.fixture
def graph_5_third_layer():
    return ThirdLayerGraph_5()


@pytest.fixture
def graph_5_second_layer(graph_5_third_layer):
    graph = SecondLayerGraph()
    graph.add_worker("func_2", graph_5_third_layer, dependencies=["start"])
    return graph


@pytest.fixture
def graph_5(graph_5_second_layer):
    graph = TopGraph_5()
    graph.add_worker("middle", graph_5_second_layer, dependencies=["start"])
    return graph


@pytest.mark.asyncio
async def test_graph_5_first_interaction(graph_5: TopGraph_5, request, db_base_path):
    """Test first human interaction in sequential layers."""
    try:
        await graph_5.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 1
        assert e.interactions[0].event.event_type == "if_add_in_third_layer"
        assert "Current value is 116" in e.interactions[0].event.data["prompt_to_user"]
        request.config.cache.set("interaction_ids_5", [e.interactions[0].interaction_id])
        (db_base_path / "graph_5.bytes").write_bytes(e.snapshot.serialized_bytes)
        (db_base_path / "graph_5.version").write_text(e.snapshot.serialization_version)


@pytest.fixture
def graph_5_deserialized_first(db_base_path):
    return deserialize_top_graph_5(db_base_path / "graph_5.bytes", db_base_path / "graph_5.version")


@pytest.mark.asyncio
async def test_graph_5_second_interaction(graph_5_deserialized_first, request, db_base_path):
    """Test second human interaction in sequential layers (top layer)."""
    interaction_ids = request.config.cache.get("interaction_ids_5", None)
    feedback = InteractionFeedback(interaction_id=interaction_ids[0], data="yes")
    try:
        await graph_5_deserialized_first.arun(feedback_data=feedback)
    except InteractionException as e:
        assert len(e.interactions) == 1
        assert e.interactions[0].event.event_type == "if_add_in_top_layer"
        assert "Current value is 1284" in e.interactions[0].event.data["prompt_to_user"]
        request.config.cache.set("interaction_ids_5", [e.interactions[0].interaction_id])
        (db_base_path / "graph_5.bytes").write_bytes(e.snapshot.serialized_bytes)
        (db_base_path / "graph_5.version").write_text(e.snapshot.serialization_version)


@pytest.fixture
def graph_5_deserialized_second(db_base_path):
    return deserialize_top_graph_5(db_base_path / "graph_5.bytes", db_base_path / "graph_5.version")


@pytest.mark.asyncio
async def test_graph_5_final(graph_5_deserialized_second, request):
    """Test final completion after sequential interactions in different layers."""
    interaction_ids = request.config.cache.get("interaction_ids_5", None)
    feedback = InteractionFeedback(interaction_id=interaction_ids[0], data="yes")
    result = await graph_5_deserialized_second.arun(feedback_data=feedback)
    assert result == 1286
