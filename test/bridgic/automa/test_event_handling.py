import pytest
import asyncio

from bridgic.automa import GraphAutoma
from bridgic.automa import worker
from bridgic.automa.worker import Worker
from bridgic.automa.interaction.event_handling import ProgressEvent, Event, Feedback, FeedbackSender

class TopGraph(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 1

    # The 'middle' worker is an Automa, which will be added by add_worker() method.

    @worker(dependencies=["middle"])
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

    @worker(dependencies=["func_1", "func_2"])
    async def end(self, x1: int, x2: int):
        return x1 + x2

class Graph_1_TestProgressEvent(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 100

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        progress_event = ProgressEvent(
            event_type="progress",
            progress=0.6,
            data=f"progress_x:{x}"
        )
        self.post_event(progress_event)
        return x + 200

    @worker(dependencies=["start"])
    async def func_2(self, x: int):
        return x + 300

    @worker(dependencies=["func_1", "func_2"])
    async def end(self, x1: int, x2: int):
        return x1 + x2

@pytest.fixture
def graph_1_third_layer():
    graph = Graph_1_TestProgressEvent(output_worker_key="end")
    return graph

@pytest.fixture
def graph_1_second_layer(graph_1_third_layer):
    graph = SecondLayerGraph(output_worker_key="end")
    graph.add_worker(
        "func_2", 
        graph_1_third_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_1(graph_1_second_layer):
    graph = TopGraph(output_worker_key="end")
    graph.add_worker(
        "middle", 
        graph_1_second_layer,
        dependencies=["start"]
    )

    def event_handler(event: Event):
        assert event.event_type == "progress"
        assert event.progress == 0.6
        assert event.data == "progress_x:116"

    graph.register_event_handler("progress", event_handler)
    return graph

@pytest.mark.asyncio
async def test_graph_1(graph_1):
    result = await graph_1.process_async(x=5)
    assert result == 770

#############################################################################

class Graph_2_TestFeedback(GraphAutoma):
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
        feedback = await self.post_event(event)
        if feedback.data == "yes":
            return x + 200
        return x

    @worker(dependencies=["start"])
    def func_2(self, x: int):
        # Test posting a progress event in a non-async method.
        progress_event = ProgressEvent(
            progress=0.7,
        )
        self.post_event(progress_event)
        return x + 300

    @worker(dependencies=["func_1", "func_2"])
    async def end(self, x1: int, x2: int):
        return x1 + x2

@pytest.fixture
def graph_2_third_layer():
    graph = Graph_2_TestFeedback(output_worker_key="end")
    return graph

@pytest.fixture
def graph_2_second_layer(graph_2_third_layer):
    graph = SecondLayerGraph(output_worker_key="end")
    graph.add_worker(
        "func_2", 
        graph_2_third_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_2(graph_2_second_layer):
    graph = TopGraph(output_worker_key="end")
    graph.add_worker(
        "middle", 
        graph_2_second_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_2_feedback_yes(graph_2):
    async def give_feedback_yes(feedback_sender: FeedbackSender):
        await asyncio.sleep(0.1)
        feedback_sender.send(Feedback(data="yes"))

    def event_handler(event: Event, feedback_sender: FeedbackSender):
        assert event.event_type == "if_add"
        assert "(yes/no)" in event.data["prompt_to_user"]

        # This simulates the application layer code. Normally, at this point, the event content would be displayed to the user, and the user would provide feedback based on the event. After the user gives feedback, typically in a different task within the same event loop, the FeedbackSender would be called. This code here provides a basic simulation of that process.
        asyncio.create_task(give_feedback_yes(feedback_sender))
    
    graph_2.register_event_handler("if_add", event_handler)
    return graph_2

@pytest.mark.asyncio
async def test_graph_2_feedback_yes(graph_2_feedback_yes):
    result = await graph_2_feedback_yes.process_async(x=5)
    assert result == 770

@pytest.fixture
def graph_2_feedback_no(graph_2):
    async def give_feedback_no(feedback_sender: FeedbackSender):
        await asyncio.sleep(0.1)
        feedback_sender.send(Feedback(data="no"))

    def event_handler_if_add(event: Event, feedback_sender: FeedbackSender):
        assert event.event_type == "if_add"
        assert "(yes/no)" in event.data["prompt_to_user"]

        # This simulates the application layer code. Normally, at this point, the event content would be displayed to the user, and the user would provide feedback based on the event. After the user gives feedback, typically in a different task within the same event loop, the FeedbackSender would be called. This code here provides a basic simulation of that process.
        asyncio.create_task(give_feedback_no(feedback_sender))
    
    graph_2.register_event_handler("if_add", event_handler_if_add)

    def event_handler_default(event: Event):
        if event.event_type is None:
            assert isinstance(event, ProgressEvent)
            assert event.progress == 0.7

    # Test the default event handler.
    graph_2.register_event_handler(None, event_handler_default)
    return graph_2

@pytest.mark.asyncio
async def test_graph_2_feedback_no(graph_2_feedback_no):
    result = await graph_2_feedback_no.process_async(x=5)
    assert result == 570

@pytest.fixture
def graph_2_feedback_no_in_same_task(graph_2):
    def event_handler(event: Event, feedback_sender: FeedbackSender):
        assert event.event_type == "if_add"
        assert "(yes/no)" in event.data["prompt_to_user"]
        # It also works to send feedback in the same task.
        feedback_sender.send(Feedback(data="no"))

    graph_2.register_event_handler("if_add", event_handler)
    return graph_2

@pytest.mark.asyncio
async def test_graph_2_feedback_no_in_same_task(graph_2_feedback_no_in_same_task):
    result = await graph_2_feedback_no_in_same_task.process_async(x=5)
    assert result == 570

@pytest.fixture
def graph_2_feedback_no_in_different_thread(graph_2):
    def give_feedback_no(feedback_sender: FeedbackSender):
        # Simulate the case that the FeedbackSender is called in a different thread.
        feedback_sender.send(Feedback(data="no"))

    def event_handler(event: Event, feedback_sender: FeedbackSender):
        assert event.event_type == "if_add"
        assert "(yes/no)" in event.data["prompt_to_user"]
        # It also works to send feedback in a different thread.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(give_feedback_no, feedback_sender)

    graph_2.register_event_handler("if_add", event_handler)
    return graph_2

@pytest.mark.asyncio
async def test_graph_2_feedback_no_in_different_thread(graph_2_feedback_no_in_different_thread):
    result = await graph_2_feedback_no_in_different_thread.process_async(x=5)
    assert result == 570

#############################################################################

class Graph_3_TestTwoSimultaneousFeedback(GraphAutoma):
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
        feedback = await self.post_event(event)
        if feedback.data == "yes":
            return x + 200
        return x

    @worker(dependencies=["start"])
    async def func_2(self, x: int):
        event = Event(
            event_type="add_x",
            data={
                "prompt_to_user": f"Current value is {x}, how much do you want to add to it? Please input a number."
            }
        )
        feedback = await self.post_event(event)
        return x + feedback.data

    @worker(dependencies=["func_1", "func_2"])
    async def end(self, x1: int, x2: int):
        return x1 + x2

@pytest.fixture
def graph_3_third_layer():
    graph = Graph_3_TestTwoSimultaneousFeedback(output_worker_key="end")
    return graph

@pytest.fixture
def graph_3_second_layer(graph_3_third_layer):
    graph = SecondLayerGraph(output_worker_key="end")
    graph.add_worker(
        "func_2", 
        graph_3_third_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_3(graph_3_second_layer):
    graph = TopGraph(output_worker_key="end")
    graph.add_worker(
        "middle", 
        graph_3_second_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_3_with_two_feedbacks(graph_3):
    async def give_feedback_no(feedback_sender: FeedbackSender):
        await asyncio.sleep(0.1)
        feedback_sender.send(Feedback(data="no"))

    def event_handler_if_add(event: Event, feedback_sender: FeedbackSender):
        assert event.event_type == "if_add"
        assert "(yes/no)" in event.data["prompt_to_user"]

        # Simulates the application layer code that runs in a different task of the same event loop.
        asyncio.create_task(give_feedback_no(feedback_sender))
    
    graph_3.register_event_handler("if_add", event_handler_if_add)

    async def give_feedback_335(feedback_sender: FeedbackSender):
        await asyncio.sleep(0.1)
        feedback_sender.send(Feedback(data=335))

    def event_handler_add_x(event: Event, feedback_sender: FeedbackSender):
        assert event.event_type == "add_x"
        assert "input a number" in event.data["prompt_to_user"]

        # Simulates the application layer code that runs in a different task of the same event loop.
        asyncio.create_task(give_feedback_335(feedback_sender))
    
    graph_3.register_event_handler("add_x", event_handler_add_x)

    return graph_3

@pytest.mark.asyncio
async def test_graph_3_two_simultaneous_feedbacks(graph_3_with_two_feedbacks):
    result = await graph_3_with_two_feedbacks.process_async(x=5)
    assert result == 770 - 200 + 35

#############################################################################

class MyWorker1(Worker):
    async def process_async(self, x: int):
        event = Event(
            event_type="if_add",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 200 to it (yes/no) ?"
            }
        )
        feedback = await self.post_event(event)
        if feedback.data == "yes":
            return x + 200
        return x

class MyWorker2(Worker):
    async def process_async(self, x: int):
        progress_event = ProgressEvent(
            progress=0.7,
        )
        self.post_event(progress_event)
        return x + 300

class Graph_4_TestFeedback_CustomWorkers(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 100

    @worker(dependencies=["worker_1", "worker_2"])
    async def end(self, x1: int, x2: int):
        return x1 + x2

@pytest.fixture
def graph_4_third_layer():
    graph = Graph_4_TestFeedback_CustomWorkers(output_worker_key="end")
    worker1 = MyWorker1()
    worker2 = MyWorker2()
    graph.add_worker("worker_1", worker1, dependencies=["start"])
    graph.add_worker("worker_2", worker2, dependencies=["start"])
    return graph

@pytest.fixture
def graph_4_second_layer(graph_4_third_layer):
    graph = SecondLayerGraph(output_worker_key="end")
    graph.add_worker(
        "func_2", 
        graph_4_third_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_4(graph_4_second_layer):
    graph = TopGraph(output_worker_key="end")
    graph.add_worker(
        "middle", 
        graph_4_second_layer,
        dependencies=["start"]
    )
    return graph

@pytest.fixture
def graph_4_feedback_no(graph_4):
    async def give_feedback_no(feedback_sender: FeedbackSender):
        await asyncio.sleep(0.1)
        feedback_sender.send(Feedback(data="no"))

    def event_handler_if_add(event: Event, feedback_sender: FeedbackSender):
        assert event.event_type == "if_add"
        assert "(yes/no)" in event.data["prompt_to_user"]

        # This simulates the application layer code. Normally, at this point, the event content would be displayed to the user, and the user would provide feedback based on the event. After the user gives feedback, typically in a different task within the same event loop, the FeedbackSender would be called. This code here provides a basic simulation of that process.
        asyncio.create_task(give_feedback_no(feedback_sender))
    
    graph_4.register_event_handler("if_add", event_handler_if_add)

    def event_handler_default(event: Event):
        if event.event_type is None:
            assert isinstance(event, ProgressEvent)
            assert event.progress == 0.7

    # Test the default event handler.
    graph_4.register_event_handler(None, event_handler_default)
    return graph_4

@pytest.mark.asyncio
async def test_graph_4_feedback_no(graph_4_feedback_no):
    result = await graph_4_feedback_no.process_async(x=5)
    assert result == 570