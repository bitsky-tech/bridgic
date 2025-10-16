import pytest
import asyncio

from bridgic.core.automa import GraphAutoma
from bridgic.core.automa import worker
from bridgic.core.automa.interaction import Event
from bridgic.core.automa.interaction import InteractionFeedback, InteractionException
from bridgic.core.automa import Snapshot

# Shared fixtures for all test cases.
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")

##### Test cases for a simple Automa && a basic human interaction process #####

class AdderAutoma1(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int):
        return x + 1

    @worker(dependencies=["func_1"], is_output=True)
    async def func_2(self, x: int):
        event = Event(
            event_type="if_add",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 200 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 200
        return x + 2

@pytest.fixture
def adder_automa1():
    return AdderAutoma1()

@pytest.mark.asyncio
async def test_adder_automa_1_interact_serialized(adder_automa1: AdderAutoma1, request, db_base_path):
    try:
        result = await adder_automa1.arun(x=100)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is 101, do you want to add another 200 to it (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("interaction_id", interaction_id)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "adder_automa1.bytes"
        version_file = db_base_path / "adder_automa1.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def deserialized_adder_automa1(db_base_path):
        bytes_file = db_base_path / "adder_automa1.bytes"
        version_file = db_base_path / "adder_automa1.version"
        serialized_bytes = bytes_file.read_bytes()
        serialization_version = version_file.read_text()
        snapshot = Snapshot(
            serialized_bytes=serialized_bytes, 
            serialization_version=serialization_version
        )
        # Snapshot is restored.
        assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
        deserialized_automa = AdderAutoma1.load_from_snapshot(snapshot)
        assert type(deserialized_automa) is AdderAutoma1
        return deserialized_automa

@pytest.fixture
def interaction_feedback_1_yes(request):
    interaction_id = request.config.cache.get("interaction_id", None)
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="yes"
    )
    return feedback

@pytest.mark.asyncio
async def test_adder_automa_1_interact_with_yes_feedback(interaction_feedback_1_yes, deserialized_adder_automa1):
    # Note: no need to pass the original argument x=100, because the deserialized_adder_automa1 is restored from the snapshot.
    result = await deserialized_adder_automa1.arun(
        interaction_feedback=interaction_feedback_1_yes
    )
    assert result == 303

@pytest.fixture
def interaction_feedback_1_no(request):
    interaction_id = request.config.cache.get("interaction_id", None)
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="no"
    )
    return feedback

@pytest.mark.asyncio
async def test_adder_automa_1_interact_with_no_feedback(interaction_feedback_1_no, deserialized_adder_automa1):
    result = await deserialized_adder_automa1.arun(
        interaction_feedback=interaction_feedback_1_no
    )
    assert result == 103

##### Test cases for a simple Automa && a human interaction process with the start worker #####

class AdderAutoma2(GraphAutoma):
    @worker(is_start=True)
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
        return x + 1

    @worker(dependencies=["func_1"], is_output=True)
    async def func_2(self, x: int):
        return x + 2

@pytest.fixture
def adder_automa2():
    return AdderAutoma2()

@pytest.mark.asyncio
async def test_adder_automa_2_interact_serialized(adder_automa2: AdderAutoma2, request, db_base_path):
    try:
        result = await adder_automa2.arun(x=100)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is 100, do you want to add another 200 to it (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("interaction_id", interaction_id)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "adder_automa2.bytes"
        version_file = db_base_path / "adder_automa2.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def deserialized_adder_automa2(db_base_path):
        bytes_file = db_base_path / "adder_automa2.bytes"
        version_file = db_base_path / "adder_automa2.version"
        serialized_bytes = bytes_file.read_bytes()
        serialization_version = version_file.read_text()
        snapshot = Snapshot(
            serialized_bytes=serialized_bytes, 
            serialization_version=serialization_version
        )
        # Snapshot is restored.
        assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
        deserialized_automa = AdderAutoma2.load_from_snapshot(snapshot)
        assert type(deserialized_automa) is AdderAutoma2
        return deserialized_automa

@pytest.fixture
def interaction_feedback_2_yes(request):
    interaction_id = request.config.cache.get("interaction_id", None)
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="yes"
    )
    return feedback

@pytest.mark.asyncio
async def test_adder_automa_2_interact_with_yes_feedback(interaction_feedback_2_yes, deserialized_adder_automa2):
    result = await deserialized_adder_automa2.arun(
        interaction_feedback=interaction_feedback_2_yes
    )
    assert result == 303

##### Test cases for nested Automas && multiple simultaneous human interactions in parallel branches #####

class TopGraph(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 1

    # The 'middle' worker is an Automa, which will be added by add_worker() method.

    @worker(dependencies=["middle"], is_output=True)
    async def end(self, x: int):
        return x + 2

class SecondLayerGraph(GraphAutoma):
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
            event_type="if_add", # event_type do not need to be unique.
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

# Utility function to deserialize a TopGraph from a bytes file and a version file.
def deserialize_top_graph(bytes_file, version_file):
    serialized_bytes = bytes_file.read_bytes()
    serialization_version = version_file.read_text()
    snapshot = Snapshot(
        serialized_bytes=serialized_bytes, 
        serialization_version=serialization_version
    )
    # Snapshot is restored.
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_graph = TopGraph.load_from_snapshot(snapshot)
    assert type(deserialized_graph) is TopGraph
    return deserialized_graph


@pytest.fixture
def graph_1_third_layer():
    graph = ThirdLayerGraph_1_ParallelBranches()
    return graph

@pytest.fixture
def graph_1_second_layer(graph_1_third_layer):
    graph = SecondLayerGraph()
    graph.add_worker(
        "func_2", 
        graph_1_third_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_1(graph_1_second_layer):
    graph = TopGraph()
    graph.add_worker(
        "middle", 
        graph_1_second_layer,
        dependencies=["start"]
    )
    return graph

@pytest.mark.asyncio
async def test_graph_1_serialized(graph_1: TopGraph, request, db_base_path):
    try:
        result = await graph_1.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 2
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add"
            assert "Current value is 116" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
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
    interaction_ids = request.config.cache.get("interaction_ids", None)
    # Mock a feedback for each interaction_id.
    feedback_no = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="no"
    )
    feedback_yes = InteractionFeedback(
        interaction_id=interaction_ids[1],
        data="yes"
    )
    # Return a list of feedbacks. The order of the feedbacks does not matter, as each feedback is uniquely identified by its interaction_id.
    return [feedback_yes, feedback_no]

@pytest.mark.asyncio
async def test_graph_1_deserialized(interaction_feedbacks_for_graph_1, graph_1_deserialized):
    result = await graph_1_deserialized.arun(
        interaction_feedbacks=interaction_feedbacks_for_graph_1
    )
    assert result == 1286 - 200

##### Test cases for nested Automas: one parallel branch is a human interaction, while the other branch is an (netsted) Automa. #####

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

    # The 'func_2' worker is an Automa, which will be added by add_worker() method.

    @worker(dependencies=["func_1", "func_2"], is_output=True)
    async def end(self, x1: int, x2: int):
        return x1 + x2

class ThirdLayerGraph_2(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 100

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        # A chance to switch to concurrent stasks...
        await asyncio.sleep(0.0)
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
    graph = ThirdLayerGraph_2()
    return graph

@pytest.fixture
def graph_2_second_layer(graph_2_third_layer):
    graph = SecondLayerGraph_2_Parallel_Interaction()
    graph.add_worker(
        "func_2", 
        graph_2_third_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_2(graph_2_second_layer):
    graph = TopGraph()
    graph.add_worker(
        "middle", 
        graph_2_second_layer,
        dependencies=["start"]
    )
    return graph

@pytest.mark.asyncio
async def test_graph_2_serialized(graph_2: TopGraph, request, db_base_path):
    try:
        result = await graph_2.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 1
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add"
            assert "Current value is 16" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_2.bytes"
        version_file = db_base_path / "graph_2.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def graph_2_deserialized(db_base_path):
        bytes_file = db_base_path / "graph_2.bytes"
        version_file = db_base_path / "graph_2.version"
        return deserialize_top_graph(bytes_file, version_file)

@pytest.mark.asyncio
async def test_graph_2_deserialized_without_feedback(graph_2_deserialized: TopGraph, request, db_base_path):
    # This case is uncommon.
    try:
        result = await graph_2_deserialized.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 1
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add"
            assert "Current value is 16" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        old_interaction_ids = request.config.cache.get("interaction_ids", None)
        assert old_interaction_ids[0] == interaction_ids[0]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_2_again.bytes"
        version_file = db_base_path / "graph_2_again.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def graph_2_deserialized_again(db_base_path):
        bytes_file = db_base_path / "graph_2_again.bytes"
        version_file = db_base_path / "graph_2_again.version"
        return deserialize_top_graph(bytes_file, version_file)


@pytest.fixture
def feedback_no_for_graph_2(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feedback_no = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="no"
    )
    return feedback_no

@pytest.fixture
def feedback_yes_for_graph_2(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feedback_yes = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="yes"
    )
    return feedback_yes

@pytest.mark.asyncio
async def test_graph_2_deserialized_again(feedback_no_for_graph_2, feedback_yes_for_graph_2, graph_2_deserialized_again):
    result = await graph_2_deserialized_again.arun(
        interaction_feedback=feedback_no_for_graph_2
    )
    assert result == 1286 - 20

    # The second feedback with the same interaction_id should be ignored.
    # Here is actually a total Automa rerun. Therefore, the input arguments must be provided once again.
    with pytest.raises(Exception, match="required positional argument"):
        result = await graph_2_deserialized_again.arun(
            interaction_feedback=feedback_yes_for_graph_2
        )

##### Test cases for nested Automas: Multiple human interactions in a worker. #####

class SecondLayerGraph_3_Multiple_Interactions(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 10

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        # The first human interaction.
        event = Event(
            event_type="if_add_1",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 20 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 20

        # The second human interaction.
        event = Event(
            event_type="if_add_2",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 30 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 30

        # The third human interaction.
        event = Event(
            event_type="if_add_3",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 40 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 40

        return x

    # The 'func_2' worker is an Automa, which will be added by add_worker() method.

    @worker(dependencies=["func_1", "func_2"], is_output=True)
    async def end(self, x1: int, x2: int):
        return x1 + x2

@pytest.fixture
def graph_3_second_layer(graph_2_third_layer):
    # Reuse graph_2_third_layer in this test case.
    graph = SecondLayerGraph_3_Multiple_Interactions()
    graph.add_worker(
        "func_2", 
        graph_2_third_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_3(graph_3_second_layer):
    graph = TopGraph()
    graph.add_worker(
        "middle", 
        graph_3_second_layer,
        dependencies=["start"]
    )
    return graph

@pytest.mark.asyncio
async def test_graph_3_serialized_first(graph_3: TopGraph, request, db_base_path):
    try:
        result = await graph_3.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 1
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add_1"
            assert "Current value is 16" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_3_first_interaction.bytes"
        version_file = db_base_path / "graph_3_first_interaction.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def graph_3_deserialized_first(db_base_path):
        bytes_file = db_base_path / "graph_3_first_interaction.bytes"
        version_file = db_base_path / "graph_3_first_interaction.version"
        return deserialize_top_graph(bytes_file, version_file)

@pytest.fixture
def feedback_yes_for_graph_3_first_interaction(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feedback_yes = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="yes"
    )
    return feedback_yes

@pytest.mark.asyncio
async def test_graph_3_deserialized_first(
    feedback_yes_for_graph_3_first_interaction,
    graph_3_deserialized_first: TopGraph, 
    request, 
    db_base_path
):
    try:
        result = await graph_3_deserialized_first.arun(
            interaction_feedback=feedback_yes_for_graph_3_first_interaction
        )
    except InteractionException as e:
        assert len(e.interactions) == 1
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add_2"
            assert "Current value is 36" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_3_second_interaction.bytes"
        version_file = db_base_path / "graph_3_second_interaction.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def graph_3_deserialized_second(db_base_path):
        bytes_file = db_base_path / "graph_3_second_interaction.bytes"
        version_file = db_base_path / "graph_3_second_interaction.version"
        return deserialize_top_graph(bytes_file, version_file)


@pytest.mark.asyncio
async def test_graph_3_deserialized_second_without_feedback(graph_3_deserialized_second: TopGraph, request, db_base_path):
    # This case is uncommon.
    try:
        result = await graph_3_deserialized_second.arun(
            interaction_feedback=None
        )
    except InteractionException as e:
        assert len(e.interactions) == 1
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add_2" # this event type again
            assert "Current value is 36" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        old_interaction_ids = request.config.cache.get("interaction_ids", None)
        assert old_interaction_ids[0] == interaction_ids[0]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_3_second_interaction.bytes"
        version_file = db_base_path / "graph_3_second_interaction.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)


@pytest.fixture
def feedback_no_for_graph_3_second_interaction(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feedback_no = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="no"
    )
    return feedback_no

@pytest.fixture
def graph_3_deserialized_second_again(db_base_path):
        bytes_file = db_base_path / "graph_3_second_interaction.bytes"
        version_file = db_base_path / "graph_3_second_interaction.version"
        return deserialize_top_graph(bytes_file, version_file)

@pytest.mark.asyncio
async def test_graph_3_deserialized_second_with_feedback(
    feedback_no_for_graph_3_second_interaction,
    graph_3_deserialized_second_again: TopGraph, 
    request, 
    db_base_path
):
    try:
        result = await graph_3_deserialized_second_again.arun(
            interaction_feedback=feedback_no_for_graph_3_second_interaction
        )
    except InteractionException as e:
        assert len(e.interactions) == 1
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add_3"
            assert "Current value is 36" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_3_third_interaction.bytes"
        version_file = db_base_path / "graph_3_third_interaction.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def feedback_yes_for_graph_3_third_interaction(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feedback_yes = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="yes"
    )
    return feedback_yes

@pytest.fixture
def graph_3_deserialized_third(db_base_path):
        bytes_file = db_base_path / "graph_3_third_interaction.bytes"
        version_file = db_base_path / "graph_3_third_interaction.version"
        return deserialize_top_graph(bytes_file, version_file)

@pytest.mark.asyncio
async def test_graph_3_deserialized_third(
    feedback_yes_for_graph_3_third_interaction,
    graph_3_deserialized_third
):
    result = await graph_3_deserialized_third.arun(
        interaction_feedback=feedback_yes_for_graph_3_third_interaction
    )
    assert result == 1286 + 40

##### Test cases for nested Automas: Simultaneous human interactions in different layers. #####

class SecondLayerGraph_4(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 10

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        event = Event(
            event_type="if_add_in_second_layer",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 20 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 20

        return x

    # The 'func_2' worker is an Automa, which will be added by add_worker() method.

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
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 300 to it (yes/no) ?"
            }
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
    graph = ThirdLayerGraph_4()
    return graph

@pytest.fixture
def graph_4_second_layer(graph_4_third_layer):
    graph = SecondLayerGraph_4()
    graph.add_worker(
        "func_2", 
        graph_4_third_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_4(graph_4_second_layer):
    graph = TopGraph()
    graph.add_worker(
        "middle", 
        graph_4_second_layer,
        dependencies=["start"]
    )
    return graph

@pytest.mark.asyncio
async def test_graph_4_serialized(graph_4: TopGraph, request, db_base_path):
    try:
        result = await graph_4.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 2
        assert e.interactions[0].event.event_type == "if_add_in_second_layer"
        assert "Current value is 16" in e.interactions[0].event.data["prompt_to_user"]
        assert e.interactions[1].event.event_type == "if_add_in_third_layer"
        assert "Current value is 116" in e.interactions[1].event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_4.bytes"
        version_file = db_base_path / "graph_4.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def graph_4_deserialized(db_base_path):
        bytes_file = db_base_path / "graph_4.bytes"
        version_file = db_base_path / "graph_4.version"
        return deserialize_top_graph(bytes_file, version_file)

@pytest.fixture
def interaction_feedbacks_for_graph_4(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    # Mock a feedback for each interaction_id.
    feedback_1 = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="yes"
    )
    feedback_2 = InteractionFeedback(
        interaction_id=interaction_ids[1],
        data="yes"
    )
    return [feedback_1, feedback_2]

@pytest.mark.asyncio
async def test_graph_4_deserialized(interaction_feedbacks_for_graph_4, graph_4_deserialized):
    result = await graph_4_deserialized.arun(
        interaction_feedbacks=interaction_feedbacks_for_graph_4
    )
    assert result == 1286

##### Test cases for nested Automas: Sequential human interactions in different layers. #####
##### In this test case, human interactions occur in the top layer and the third layer. #####

class TopGraph_5(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 1

    # The 'middle' worker is an Automa, which will be added by add_worker() method.

    @worker(dependencies=["middle"], is_output=True)
    async def end(self, x: int):
        event = Event(
            event_type="if_add_in_top_layer",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 2 to it (yes/no) ?"
            }
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
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 300 to it (yes/no) ?"
            }
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
def graph_5_third_layer():
    graph = ThirdLayerGraph_5()
    return graph

@pytest.fixture
def graph_5_second_layer(graph_5_third_layer):
    graph = SecondLayerGraph()
    graph.add_worker(
        "func_2", 
        graph_5_third_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_5(graph_5_second_layer):
    graph = TopGraph_5()
    graph.add_worker(
        "middle", 
        graph_5_second_layer,
        dependencies=["start"]
    )
    return graph

@pytest.mark.asyncio
async def test_graph_5_serialized(graph_5: TopGraph_5, request, db_base_path):
    try:
        result = await graph_5.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 1
        assert e.interactions[0].event.event_type == "if_add_in_third_layer"
        assert "Current value is 116" in e.interactions[0].event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_5_first.bytes"
        version_file = db_base_path / "graph_5_first.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def feedback_yes_for_graph_5_interaction_in_third_layer(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feedback_yes = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="yes"
    )
    return feedback_yes

@pytest.fixture
def graph_5_deserialized_first(db_base_path):
        bytes_file = db_base_path / "graph_5_first.bytes"
        version_file = db_base_path / "graph_5_first.version"
        serialized_bytes = bytes_file.read_bytes()
        serialization_version = version_file.read_text()
        snapshot = Snapshot(
            serialized_bytes=serialized_bytes, 
            serialization_version=serialization_version
        )
        # Snapshot is restored.
        assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
        deserialized_graph = TopGraph_5.load_from_snapshot(snapshot)
        assert type(deserialized_graph) is TopGraph_5
        return deserialized_graph

@pytest.mark.asyncio
async def test_graph_5_deserialized_first(
    feedback_yes_for_graph_5_interaction_in_third_layer,
    graph_5_deserialized_first: TopGraph, 
    request, 
    db_base_path
):
    try:
        result = await graph_5_deserialized_first.arun(
            interaction_feedback=feedback_yes_for_graph_5_interaction_in_third_layer
        )
    except InteractionException as e:
        assert len(e.interactions) == 1
        assert e.interactions[0].event.event_type == "if_add_in_top_layer"
        assert "Current value is 1284" in e.interactions[0].event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_5_second.bytes"
        version_file = db_base_path / "graph_5_second.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def feedback_yes_for_graph_5_interaction_in_top_layer(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feedback_yes = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="yes"
    )
    return feedback_yes

@pytest.fixture
def graph_5_deserialized_second(db_base_path):
        bytes_file = db_base_path / "graph_5_second.bytes"
        version_file = db_base_path / "graph_5_second.version"
        serialized_bytes = bytes_file.read_bytes()
        serialization_version = version_file.read_text()
        snapshot = Snapshot(
            serialized_bytes=serialized_bytes, 
            serialization_version=serialization_version
        )
        # Snapshot is restored.
        assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
        deserialized_graph = TopGraph_5.load_from_snapshot(snapshot)
        assert type(deserialized_graph) is TopGraph_5
        return deserialized_graph

@pytest.mark.asyncio
async def test_graph_5_deserialized_second(
    feedback_yes_for_graph_5_interaction_in_top_layer, 
    graph_5_deserialized_second
):
    result = await graph_5_deserialized_second.arun(
        interaction_feedback=feedback_yes_for_graph_5_interaction_in_top_layer
    )
    assert result == 1286
