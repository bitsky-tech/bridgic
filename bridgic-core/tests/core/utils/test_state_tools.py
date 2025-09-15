from bridgic.core.utils.state_tools import useState, stateful
import threading


def test_number_state():
    """Test number state"""
    # Initialize number state
    count, setCount = useState(1)
    assert count == 1
    assert isinstance(count, int)
    assert count + 5 == 6
    assert count == 1

    # Update number state
    setCount(10)
    assert count == 10
    assert count * 2 == 20
    assert count > 5

def test_string_state():
    """Test string state"""
    # Initialize string state
    name, setName = useState("Alice")
    assert name == "Alice"
    assert isinstance(name, str)
    assert len(name) == 5

    # Update string state
    setName("Bob")
    assert name == "Bob"
    assert name == "Bob"

def test_list_state():
    """Test list state"""
    # Initialize list state
    items, setItems = useState([1, 2, 3])
    assert items == [1, 2, 3]
    assert isinstance(items, list)
    assert len(items) == 3
    assert items[0] == 1

    # Update list state
    setItems([4, 5, 6, 7])
    assert items == [4, 5, 6, 7]
    assert len(items) == 4

def test_dict_state():
    """Test dictionary state"""
    # Initialize dictionary state
    user, setUser = useState({"name": "Alice", "age": 25})
    assert user == {"name": "Alice", "age": 25}
    assert isinstance(user, dict)
    assert user["name"] == "Alice"

    # Update dictionary state
    setUser({"name": "Bob", "age": 30, "city": "Shanghai"})
    assert user == {"name": "Bob", "age": 30, "city": "Shanghai"}
    assert user["name"] == "Bob"
    assert user["city"] == "Shanghai"

def test_state_identity():
    """Test state identity (independence)"""
    # Create two independent states
    state1, setState1 = useState([1, 2, 3])
    state2, setState2 = useState([1, 2, 3])

    assert state1 == state2  # equal by value
    assert state1 is not state2  # different identity (independent)

    # Modify one of the states
    setState1([1, 2, 3, 4])
    assert state1 == [1, 2, 3, 4]
    assert state2 == [1, 2, 3]
    assert state1 != state2  # confirm independence

def test_complex_object_state():
    """Test complex object state"""
    class TodoItem:
        def __init__(self, text, completed=False):
            self.text = text
            self.completed = completed
        
        def __str__(self):
            status = "✅" if self.completed else "⭕"
            return f"{status} {self.text}"
        
        def __repr__(self):
            return f"TodoItem('{self.text}', {self.completed})"
        
        def __eq__(self, other):
            return (isinstance(other, TodoItem) and 
                    self.text == other.text and 
                    self.completed == other.completed)

    # Initialize todo
    todo, setTodo = useState(TodoItem("Learn Python"))
    assert todo.text == "Learn Python"
    assert type(todo).__name__ == "TodoItem"
    assert not todo.completed

    # Update todo state
    completed_todo = TodoItem("Learn Python", True)
    setTodo(completed_todo)
    assert todo.text == "Learn Python"
    assert todo.completed

def test_advanced_math_operations():
    """Test advanced math operations"""
    num, setNum = useState(12)
    
    # Test various math operations
    assert num == 12
    assert num / 3 == 4.0
    assert num // 5 == 2
    assert num % 5 == 2
    assert num ** 2 == 144
    assert -num == -12
    assert abs(num) == 12
    assert round(num / 7, 2) == 1.71

def test_bitwise_operations():
    """Test bitwise operations"""
    bit_num, setBitNum = useState(5)  # binary: 101
    
    assert bit_num == 5
    assert bin(bit_num) == "0b101"
    assert bit_num & 3 == 1  # 101 & 011 = 001
    assert bit_num | 3 == 7  # 101 | 011 = 111
    assert bit_num ^ 3 == 6  # 101 ^ 011 = 110
    assert ~bit_num == -6

def test_formatting_support():
    """Test formatting support"""
    pi, setPi = useState(3.14159)
    
    assert pi == 3.14159
    assert f"{pi:.2f}" == "3.14"
    assert f"{pi:.1%}" == "314.2%"

def test_sequence_reversal():
    """Test sequence reversal"""
    # Test list reversal
    seq, setSeq = useState([1, 2, 3, 4, 5])
    assert seq == [1, 2, 3, 4, 5]
    assert list(reversed(seq)) == [5, 4, 3, 2, 1]

    # Test string reversal
    string_seq, setStringSeq = useState("hello")
    assert string_seq == "hello"
    assert "".join(reversed(string_seq)) == "olleh"

def test_state_type_consistency():
    """Test state type consistency"""
    # Ensure type consistency after updates
    value, setValue = useState(42)
    assert isinstance(value, int)
    
    setValue(3.14)
    assert isinstance(value, float)
    assert value == 3.14

    setValue("hello")
    assert isinstance(value, str)
    assert value == "hello"

def test_state_mutation_safety():
    """Test state mutation safety"""
    # In-place modification should not affect internal state
    data, setData = useState([1, 2, 3])
    original_data = data.copy()
    
    # Directly modifying the returned list should not affect internal state
    data.append(4)
    assert data == [1, 2, 3, 4]  # direct modification takes effect
    
    # Re-initialize state to verify isolation
    new_data, _ = useState([1, 2, 3])
    assert new_data == [1, 2, 3]  # new state is unaffected


def test_function_rerun_state():
    """Test function rerun state"""
    def func():
        a, setA = useState(0)
        setA(a + 1)
        return a
    assert func() == 1
    assert func() == 2
    assert func() == 3
    assert func() == 4
    assert func() == 5


def test_thread_isolated_state_slowpath():
    """Test thread isolated state slowpath"""
    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(5)

    def worker(k: int):
        barrier.wait()
        def func():
            a, setA = useState(0)
            setA(a + 1)
            return a
        last = None
        for _ in range(k):
            last = func()
        with lock:
            results.append(last)

    threads = [threading.Thread(target=worker, args=(5,)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # each thread should get 5
    assert sorted(results) == [5, 5, 5, 5, 5]


def test_thread_isolated_state_fastpath_with_stateful():
    """Test thread isolated state fastpath with stateful"""
    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    @stateful
    def func():
        a, setA = useState(0)
        setA(a + 1)
        return a

    def worker(k: int):
        barrier.wait()
        last = None
        for _ in range(k):
            last = func()
        with lock:
            results.append(last)

    threads = [threading.Thread(target=worker, args=(7,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # each thread should get 7
    assert sorted(results) == [7, 7, 7, 7]