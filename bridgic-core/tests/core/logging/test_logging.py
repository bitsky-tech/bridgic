"""
Tests for Bridgic Core logging functionality.
"""

import json
import logging
from io import StringIO
from unittest.mock import Mock, patch

from bridgic.core.logging import (
    get_automa_logger,
    get_event_logger,
    get_logger,
    get_model_logger,
    get_trace_logger,
    get_worker_logger,
    setup_logging,
    setup_development_logging,
    setup_production_logging,
    setup_quiet_logging,
    setup_verbose_logging,
)

from bridgic.core.logging import (
    AutomaEndEvent,
    AutomaErrorEvent,
    AutomaStartEvent,
    CacheHitEvent,
    CacheMissEvent,
    CacheSetEvent,
    CacheDeleteEvent,
    EventType,
    ModelCallEvent,
    WorkerEndEvent,
    WorkerErrorEvent,
    WorkerStartEvent,
)


class TestEventClasses:
    """Test event class functionality."""
    
    def test_worker_start_event(self):
        """Test WorkerStartEvent creation and serialization."""
        event = WorkerStartEvent(
            worker_id="worker_001",
            worker_name="TestWorker",
            input_data={"key": "value"},
            execution_id="exec_123",
            metadata={"version": "1.0"}
        )
        
        assert event.worker_id == "worker_001"
        assert event.worker_name == "TestWorker"
        assert event.input_data == {"key": "value"}
        assert event.execution_id == "exec_123"
        assert event.metadata == {"version": "1.0"}
        assert event.event_type == EventType.WORKER_START
        
        # Test JSON serialization
        event_json = str(event)
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "WorkerStart"
        assert event_data["worker_id"] == "worker_001"
    
    def test_worker_end_event(self):
        """Test WorkerEndEvent creation and serialization."""
        event = WorkerEndEvent(
            worker_id="worker_001",
            worker_name="TestWorker",
            output_data={"result": "success"},
            execution_id="exec_123",
            execution_time=1.5,
            metadata={"status": "completed"}
        )
        
        assert event.worker_id == "worker_001"
        assert event.output_data == {"result": "success"}
        assert event.execution_time == 1.5
        assert event.event_type == EventType.WORKER_END
        
        # Test JSON serialization
        event_json = str(event)
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "WorkerEnd"
        assert event_data["execution_time"] == 1.5
    
    def test_worker_error_event(self):
        """Test WorkerErrorEvent creation and serialization."""
        event = WorkerErrorEvent(
            worker_id="worker_001",
            worker_name="TestWorker",
            error_type="ValueError",
            error_message="Invalid input",
            execution_id="exec_123",
            error_traceback="Traceback...",
            metadata={"context": "test"}
        )
        
        assert event.worker_id == "worker_001"
        assert event.error_type == "ValueError"
        assert event.error_message == "Invalid input"
        assert event.event_type == EventType.WORKER_ERROR
        
        # Test JSON serialization
        event_json = str(event)
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "WorkerError"
        assert event_data["error_type"] == "ValueError"
    
    def test_automa_start_event(self):
        """Test AutomaStartEvent creation and serialization."""
        event = AutomaStartEvent(
            automa_id="automa_001",
            automa_name="TestAutoma",
            workflow_data={"steps": ["step1", "step2"]},
            execution_id="exec_456",
            metadata={"version": "2.0"}
        )
        
        assert event.automa_id == "automa_001"
        assert event.automa_name == "TestAutoma"
        assert event.workflow_data == {"steps": ["step1", "step2"]}
        assert event.event_type == EventType.AUTOMA_START
        
        # Test JSON serialization
        event_json = str(event)
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "AutomaStart"
        assert event_data["automa_id"] == "automa_001"
    
    def test_model_call_event(self):
        """Test ModelCallEvent creation and serialization."""
        event = ModelCallEvent(
            model_name="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model_response={"content": "Generated text"},
            execution_time=2.5,
            metadata={"temperature": 0.7}
        )
        
        assert event.model_name == "gpt-4"
        assert event.prompt_tokens == 100
        assert event.completion_tokens == 50
        assert event.total_tokens == 150
        assert event.event_type == EventType.MODEL_CALL
        
        # Test JSON serialization
        event_json = str(event)
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "ModelCall"
        assert event_data["model_name"] == "gpt-4"
    
    
    def test_cache_hit_event(self):
        """Test CacheHitEvent creation and serialization."""
        event = CacheHitEvent(
            cache_key="user_123",
            metadata={"cache_size": 1000}
        )
        
        assert event.cache_key == "user_123"
        assert event.event_type == EventType.CACHE_HIT
        
        # Test JSON serialization
        event_json = str(event)
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "CacheHit"
        assert event_data["cache_key"] == "user_123"
    
    def test_cache_miss_event(self):
        """Test CacheMissEvent creation and serialization."""
        event = CacheMissEvent(
            cache_key="user_456",
            metadata={"cache_size": 1000}
        )
        
        assert event.cache_key == "user_456"
        assert event.event_type == EventType.CACHE_MISS
        
        # Test JSON serialization
        event_json = str(event)
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "CacheMiss"
        assert event_data["cache_key"] == "user_456"
    
    def test_cache_set_event(self):
        """Test CacheSetEvent creation and serialization."""
        event = CacheSetEvent(
            cache_key="user_789",
            metadata={"cache_size": 1001}
        )
        
        assert event.cache_key == "user_789"
        assert event.event_type == EventType.CACHE_SET
        
        # Test JSON serialization
        event_json = str(event)
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "CacheSet"
        assert event_data["cache_key"] == "user_789"
    
    def test_cache_delete_event(self):
        """Test CacheDeleteEvent creation and serialization."""
        event = CacheDeleteEvent(
            cache_key="*",
            metadata={"operation": "clear"}
        )
        
        assert event.cache_key == "*"
        assert event.event_type == EventType.CACHE_DELETE
        
        # Test JSON serialization
        event_json = str(event)
        event_data = json.loads(event_json)
        assert event_data["event_type"] == "CacheDelete"
        assert event_data["cache_key"] == "*"


class TestBridgicLogger:
    """Test BridgicLogger functionality."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        logger = get_logger("test_component", source="TestSource")
        assert logger._source == "TestSource"
        assert logger._logger.name == "test_component"
    
    def test_worker_logger_creation(self):
        """Test worker logger creation."""
        logger = get_worker_logger("test_worker", source="TestSource")
        assert logger._source == "TestSource"
        assert "workers" in logger._logger.name
    
    def test_automa_logger_creation(self):
        """Test automa logger creation."""
        logger = get_automa_logger("test_automa", source="TestSource")
        assert logger._source == "TestSource"
        assert "automa" in logger._logger.name
    
    def test_model_logger_creation(self):
        """Test model logger creation."""
        logger = get_model_logger("test_model", source="TestSource")
        assert logger._source == "TestSource"
        assert "model" in logger._logger.name
    
    def test_event_logger_creation(self):
        """Test event logger creation."""
        logger = get_event_logger("TestSource")
        assert logger._source == "TestSource"
        assert "events" in logger._logger.name
    
    def test_trace_logger_creation(self):
        """Test trace logger creation."""
        logger = get_trace_logger("TestSource")
        assert logger._source == "TestSource"
        assert "trace" in logger._logger.name


class TestLoggingMethods:
    """Test logging method functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.logger = get_logger("test_component", source="TestSource")
        # Use a StringIO handler to capture log output
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setLevel(logging.DEBUG)
        self.logger._logger.addHandler(self.handler)
        self.logger._logger.setLevel(logging.DEBUG)
    
    def test_log_worker_start(self):
        """Test worker start logging."""
        self.logger.log_worker_start(
            worker_id="worker_001",
            worker_name="TestWorker",
            input_data={"key": "value"},
            execution_id="exec_123",
            metadata={"version": "1.0"}
        )
        
        # Verify log output
        log_output = self.log_stream.getvalue()
        assert "WorkerStart" in log_output
        assert "worker_001" in log_output
        assert "TestWorker" in log_output
    
    def test_log_worker_end(self):
        """Test worker end logging."""
        self.logger.log_worker_end(
            worker_id="worker_001",
            worker_name="TestWorker",
            output_data={"result": "success"},
            execution_id="exec_123",
            execution_time=1.5,
            metadata={"status": "completed"}
        )
        
        # Verify log output
        log_output = self.log_stream.getvalue()
        assert "WorkerEnd" in log_output
        assert "worker_001" in log_output
    
    def test_log_worker_error(self):
        """Test worker error logging."""
        self.logger.log_worker_error(
            worker_id="worker_001",
            worker_name="TestWorker",
            error_type="ValueError",
            error_message="Invalid input",
            execution_id="exec_123",
            error_traceback="Traceback...",
            metadata={"context": "test"}
        )
        
        # Verify log output
        log_output = self.log_stream.getvalue()
        assert "WorkerError" in log_output
        assert "ValueError" in log_output
    
    def test_log_automa_start(self):
        """Test automa start logging."""
        self.logger.log_automa_start(
            automa_id="automa_001",
            automa_name="TestAutoma",
            workflow_data={"steps": ["step1", "step2"]},
            execution_id="exec_456",
            metadata={"version": "2.0"}
        )
        
        # Verify log output
        log_output = self.log_stream.getvalue()
        assert "AutomaStart" in log_output
        assert "automa_001" in log_output
    
    def test_log_model_call(self):
        """Test model call logging."""
        self.logger.log_model_call(
            model_name="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            model_response={"content": "Generated text"},
            execution_time=2.5,
            metadata={"temperature": 0.7}
        )
        
        # Verify log output
        log_output = self.log_stream.getvalue()
        assert "ModelCall" in log_output
        assert "gpt-4" in log_output
    
    
    def test_log_cache_hit(self):
        """Test cache hit logging."""
        self.logger.log_cache_hit(
            cache_key="user_123",
            metadata={"cache_size": 1000}
        )
        
        # Verify log output
        log_output = self.log_stream.getvalue()
        assert "CacheHit" in log_output
        assert "user_123" in log_output
    
    def test_log_cache_miss(self):
        """Test cache miss logging."""
        self.logger.log_cache_miss(
            cache_key="user_456",
            metadata={"cache_size": 1000}
        )
        
        # Verify log output
        log_output = self.log_stream.getvalue()
        assert "CacheMiss" in log_output
        assert "user_456" in log_output
    
    def test_log_cache_set(self):
        """Test cache set logging."""
        self.logger.log_cache_set(
            cache_key="user_789",
            metadata={"cache_size": 1001}
        )
        
        # Verify log output
        log_output = self.log_stream.getvalue()
        assert "CacheSet" in log_output
        assert "user_789" in log_output
    
    def test_log_cache_delete(self):
        """Test cache delete logging."""
        self.logger.log_cache_delete(
            cache_key="*",
            metadata={"operation": "clear"}
        )
        
        # Verify log output
        log_output = self.log_stream.getvalue()
        assert "CacheDelete" in log_output
    
    def test_trace_logging(self):
        """Test trace logging methods."""
        # Test trace method
        self.logger.trace("Trace message")
        log_output = self.log_stream.getvalue()
        assert "Trace message" in log_output
        
        # Test debug method
        self.logger.debug("Debug message")
        log_output = self.log_stream.getvalue()
        assert "Debug message" in log_output
        
        # Test info method
        self.logger.info("Info message")
        log_output = self.log_stream.getvalue()
        assert "Info message" in log_output


class TestSetupLogging:
    """Test logging setup functionality."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            setup_logging()
            
            # Verify logger configuration
            assert mock_logger.setLevel.called
            assert mock_logger.addHandler.called
    
    def test_setup_logging_custom(self):
        """Test custom logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            custom_handler = Mock()
            setup_logging(
                level=logging.DEBUG,
                enable_trace=True,
                enable_events=True,
                handlers=[custom_handler]
            )
            
            # Verify logger configuration
            assert mock_logger.setLevel.called
            assert mock_logger.addHandler.called
    
    def test_setup_development_logging(self):
        """Test development logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            setup_development_logging()
            
            # Verify logger configuration
            assert mock_logger.setLevel.called
            assert mock_logger.addHandler.called
    
    def test_setup_production_logging(self):
        """Test production logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            setup_production_logging()
            
            # Verify logger configuration
            assert mock_logger.setLevel.called
            assert mock_logger.addHandler.called
    
    def test_setup_quiet_logging(self):
        """Test quiet logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            setup_quiet_logging()
            
            # Verify logger configuration
            assert mock_logger.setLevel.called
            assert mock_logger.addHandler.called
    
    def test_setup_verbose_logging(self):
        """Test verbose logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            setup_verbose_logging()
            
            # Verify logger configuration
            assert mock_logger.setLevel.called
            assert mock_logger.addHandler.called


class TestLoggingLevelsBehavior:
    """Test that logging levels control what is emitted to the terminal (handlers)."""

    def _make_stream_handler(self, level=logging.DEBUG):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        return stream, handler

    def test_worker_logger_info_visible_debug_suppressed_with_info_root(self):
        # Arrange: root INFO, custom handler at INFO
        stream, handler = self._make_stream_handler(level=logging.INFO)
        setup_logging(level=logging.INFO, enable_workers=True, handlers=[handler])

        # Act
        logger = get_worker_logger("level_test")._logger
        logger.debug("DBG message")
        logger.info("INFO message")

        # Assert
        output = handler.stream.getvalue()
        assert "INFO message" in output
        assert "DBG message" not in output

    def test_component_override_debug_visible_with_debug_handler(self):
        # Arrange: root WARNING, but we attach DEBUG handler; workers set to DEBUG
        stream, handler = self._make_stream_handler(level=logging.DEBUG)
        setup_logging(
            level=logging.WARNING,
            enable_workers=True,
            handlers=[handler],
            component_levels={"workers": logging.DEBUG},
        )

        # Act
        logger = get_worker_logger("level_test2")._logger
        logger.debug("DBG2 message")
        logger.info("INFO2 message")

        # Assert: both should appear due to DEBUG handler and workers DEBUG level
        output = handler.stream.getvalue()
        assert "DBG2 message" in output
        assert "INFO2 message" in output

    def test_quiet_setup_suppresses_worker_info(self):
        # Arrange quiet setup (workers disabled). Then attach our own handler for observation
        setup_quiet_logging()
        stream, handler = self._make_stream_handler(level=logging.DEBUG)
        # Attach handler to workers root namespace so we can capture messages if any
        workers_logger = logging.getLogger("bridgic.workers")
        workers_logger.addHandler(handler)
        workers_logger.setLevel(logging.DEBUG)

        # Act: emit via a worker logger
        logger = get_worker_logger("quiet_test")._logger
        logger.info("QUIET INFO")
        logger.debug("QUIET DBG")

        # Assert: Nothing should pass because worker loggers are effectively disabled in quiet setup
        output = handler.stream.getvalue()
        assert output.strip() == ""


class TestEventSerialization:
    """Test event serialization functionality."""
    
    def test_event_json_serialization(self):
        """Test that events can be serialized to JSON."""
        event = WorkerStartEvent(
            worker_id="worker_001",
            worker_name="TestWorker",
            input_data={"key": "value"},
            execution_id="exec_123"
        )
        
        # Test JSON serialization
        event_json = event.to_json()
        assert isinstance(event_json, str)
        
        # Test that JSON can be parsed
        event_data = json.loads(event_json)
        assert event_data["worker_id"] == "worker_001"
        assert event_data["worker_name"] == "TestWorker"
        assert event_data["event_type"] == "WorkerStart"
    
    def test_event_string_serialization(self):
        """Test that events can be serialized to string."""
        event = ModelCallEvent(
            model_name="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model_response={"content": "Generated text"}
        )
        
        # Test string serialization
        event_str = str(event)
        assert isinstance(event_str, str)
        
        # Test that string can be parsed as JSON
        event_data = json.loads(event_str)
        assert event_data["model_name"] == "gpt-4"
        assert event_data["event_type"] == "ModelCall"

