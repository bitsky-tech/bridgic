"""
Main logging module for Bridgic Core.

This module provides convenient logging functionality for the Bridgic framework,
including structured event logging and trace logging.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from bridgic.core.constants import (
    AUTOMA_LOGGER_NAME,
    EVENT_LOGGER_NAME,
    MODEL_LOGGER_NAME,
    ROOT_LOGGER_NAME,
    TRACE_LOGGER_NAME,
    WORKER_LOGGER_NAME,
)

from ._log_event import (
    AutomaEndEvent,
    AutomaErrorEvent,
    AutomaStartEvent,
    BridgicEvent,
    CacheHitEvent,
    CacheMissEvent,
    CacheSetEvent,
    CacheDeleteEvent,
    ModelCallEvent,
    WorkerEndEvent,
    WorkerErrorEvent,
    WorkerStartEvent,
    WorkerFerryEvent,
)


class BridgicLogger:
    """Main logger class for Bridgic framework."""
    
    def __init__(self, name: str, source: Optional[str] = None):
        """Initialize the logger.
        
        Parameters
        ----------
        name : str
            Logger name
        source : str, optional
            Source component name
        """
        self._logger = logging.getLogger(name)
        self._source = source
    
    def _log_event(self, event: BridgicEvent) -> None:
        """Log a structured event.
        
        Parameters
        ----------
        event : BridgicEvent
            The structured event to log
        """
        if self._source and hasattr(event, 'source'):
            event.source = self._source
        self._logger.debug(event)
    
    # Worker events
    def log_worker_start(
        self,
        worker_id: str,
        worker_name: str,
        input_data: Dict[str, Any],
        execution_id: Optional[str] = None,
        worker_key: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        triggered_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log worker start event.
        
        Parameters
        ----------
        worker_id : str
            Unique identifier for the worker
        worker_name : str
            Name of the worker
        input_data : Dict[str, Any]
            Input data passed to the worker
        execution_id : str, optional
            Identifier for the execution context
        worker_key : str, optional
            Key of the worker in the automa graph
        dependencies : List[str], optional
            List of worker keys this worker depends on
        triggered_by : str, optional
            Worker key that triggered this execution
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = WorkerStartEvent(
            worker_id=worker_id,
            worker_name=worker_name,
            input_data=input_data,
            execution_id=execution_id,
            worker_key=worker_key,
            dependencies=dependencies,
            triggered_by=triggered_by,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    def log_worker_end(
        self,
        worker_id: str,
        worker_name: str,
        output_data: Dict[str, Any],
        execution_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        worker_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log worker end event.
        
        Parameters
        ----------
        worker_id : str
            Unique identifier for the worker
        worker_name : str
            Name of the worker
        output_data : Dict[str, Any]
            Output data produced by the worker
        execution_id : str, optional
            Identifier for the execution context
        execution_time : float, optional
            Time taken to execute the worker in seconds
        worker_key : str, optional
            Key of the worker in the automa graph
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = WorkerEndEvent(
            worker_id=worker_id,
            worker_name=worker_name,
            output_data=output_data,
            execution_id=execution_id,
            execution_time=execution_time,
            worker_key=worker_key,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    def log_worker_error(
        self,
        worker_id: str,
        worker_name: str,
        error_type: str,
        error_message: str,
        execution_id: Optional[str] = None,
        error_traceback: Optional[str] = None,
        execution_time: Optional[float] = None,
        worker_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log worker error event.
        
        Parameters
        ----------
        worker_id : str
            Unique identifier for the worker
        worker_name : str
            Name of the worker
        error_type : str
            Type of the error
        error_message : str
            Error message
        execution_id : str, optional
            Identifier for the execution context
        error_traceback : str, optional
            Error traceback if available
        execution_time : float, optional
            Time taken before the error occurred in seconds
        worker_key : str, optional
            Key of the worker in the automa graph
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = WorkerErrorEvent(
            worker_id=worker_id,
            worker_name=worker_name,
            error_type=error_type,
            error_message=error_message,
            execution_id=execution_id,
            error_traceback=error_traceback,
            execution_time=execution_time,
            worker_key=worker_key,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    # Automa events
    def log_automa_start(
        self,
        automa_id: str,
        automa_name: str,
        workflow_data: Dict[str, Any],
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log automa start event.
        
        Parameters
        ----------
        automa_id : str
            Unique identifier for the automa
        automa_name : str
            Name of the automa
        workflow_data : Dict[str, Any]
            Workflow data for the automa
        execution_id : str, optional
            Identifier for the execution context
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = AutomaStartEvent(
            automa_id=automa_id,
            automa_name=automa_name,
            workflow_data=workflow_data,
            execution_id=execution_id,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    def log_automa_end(
        self,
        automa_id: str,
        automa_name: str,
        result_data: Dict[str, Any],
        execution_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log automa end event.
        
        Parameters
        ----------
        automa_id : str
            Unique identifier for the automa
        automa_name : str
            Name of the automa
        result_data : Dict[str, Any]
            Result data produced by the automa
        execution_id : str, optional
            Identifier for the execution context
        execution_time : float, optional
            Time taken to execute the automa in seconds
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = AutomaEndEvent(
            automa_id=automa_id,
            automa_name=automa_name,
            result_data=result_data,
            execution_id=execution_id,
            execution_time=execution_time,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    def log_automa_error(
        self,
        automa_id: str,
        automa_name: str,
        error_type: str,
        error_message: str,
        execution_id: Optional[str] = None,
        error_traceback: Optional[str] = None,
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log automa error event.
        
        Parameters
        ----------
        automa_id : str
            Unique identifier for the automa
        automa_name : str
            Name of the automa
        error_type : str
            Type of the error
        error_message : str
            Error message
        execution_id : str, optional
            Identifier for the execution context
        error_traceback : str, optional
            Error traceback if available
        execution_time : float, optional
            Time taken before the error occurred in seconds
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = AutomaErrorEvent(
            automa_id=automa_id,
            automa_name=automa_name,
            error_type=error_type,
            error_message=error_message,
            execution_id=execution_id,
            error_traceback=error_traceback,
            execution_time=execution_time,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    def log_worker_ferry(
        self,
        worker_id: str,
        worker_name: str,
        from_worker_key: str,
        to_worker_key: str,
        ferry_data: Optional[Dict[str, Any]] = None,
        automa_name: Optional[str] = None,
        automa_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        worker_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log worker ferry_to event.
        
        Parameters
        ----------
        worker_id : str
            Unique identifier for the worker
        worker_name : str
            Name of the worker
        from_worker_key : str
            Key of the worker initiating the ferry
        to_worker_key : str
            Key of the target worker
        ferry_data : Dict[str, Any], optional
            Data being passed in the ferry operation
        automa_name : str, optional
            Name of the automa containing these workers
        automa_id : str, optional
            ID of the automa containing these workers
        execution_id : str, optional
            Identifier for the execution context
        worker_key : str, optional
            Key of the worker in the automa graph
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = WorkerFerryEvent(
            worker_id=worker_id,
            worker_name=worker_name,
            from_worker_key=from_worker_key,
            to_worker_key=to_worker_key,
            ferry_data=ferry_data or {},
            automa_name=automa_name,
            automa_id=automa_id,
            execution_id=execution_id,
            worker_key=worker_key,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    # Model events
    def log_model_call(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model_response: Dict[str, Any],
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log model call event.
        
        Parameters
        ----------
        model_name : str
            Name of the model being called
        prompt_tokens : int
            Number of tokens in the prompt
        completion_tokens : int
            Number of tokens in the completion
        model_response : Dict[str, Any]
            Response from the model
        execution_time : float, optional
            Time taken for the model call in seconds
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = ModelCallEvent(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model_response=model_response,
            execution_time=execution_time,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    
    # Cache events
    def log_cache_hit(
        self,
        cache_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log cache hit event.
        
        Parameters
        ----------
        cache_key : str
            Cache key that was hit
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = CacheHitEvent(
            cache_key=cache_key,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    def log_cache_miss(
        self,
        cache_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log cache miss event.
        
        Parameters
        ----------
        cache_key : str
            Cache key that was missed
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = CacheMissEvent(
            cache_key=cache_key,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    def log_cache_set(
        self,
        cache_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log cache set event.
        
        Parameters
        ----------
        cache_key : str
            Cache key that was set
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = CacheSetEvent(
            cache_key=cache_key,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    def log_cache_delete(
        self,
        cache_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log cache delete event.
        
        Parameters
        ----------
        cache_key : str
            Cache key that was deleted
        metadata : Dict[str, Any], optional
            Additional metadata about the event
        """
        event = CacheDeleteEvent(
            cache_key=cache_key,
            metadata=metadata or {},
        )
        self._log_event(event)
    
    # Trace logging methods
    def trace(self, message: str, *args, **kwargs) -> None:
        """Log a trace message.
        
        Parameters
        ----------
        message : str
            The trace message to log
        *args
            Additional arguments for message formatting
        **kwargs
            Additional keyword arguments for message formatting
        """
        self._logger.debug(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message.
        
        Parameters
        ----------
        message : str
            The debug message to log
        *args
            Additional arguments for message formatting
        **kwargs
            Additional keyword arguments for message formatting
        """
        self._logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log an info message.
        
        Parameters
        ----------
        message : str
            The info message to log
        *args
            Additional arguments for message formatting
        **kwargs
            Additional keyword arguments for message formatting
        """
        self._logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message.
        
        Parameters
        ----------
        message : str
            The warning message to log
        *args
            Additional arguments for message formatting
        **kwargs
            Additional keyword arguments for message formatting
        """
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log an error message.
        
        Parameters
        ----------
        message : str
            The error message to log
        *args
            Additional arguments for message formatting
        **kwargs
            Additional keyword arguments for message formatting
        """
        self._logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log a critical message.
        
        Parameters
        ----------
        message : str
            The critical message to log
        *args
            Additional arguments for message formatting
        **kwargs
            Additional keyword arguments for message formatting
        """
        self._logger.critical(message, *args, **kwargs)


def get_logger(name: str, source: Optional[str] = None) -> BridgicLogger:
    """Get a Bridgic logger instance.
    
    Parameters
    ----------
    name : str
        Logger name
    source : str, optional
        Source component name
        
    Returns
    -------
    BridgicLogger
        BridgicLogger instance
    """
    return BridgicLogger(name, source)


def get_worker_logger(worker_name: str, source: Optional[str] = None) -> BridgicLogger:
    """Get a worker logger instance.
    
    Parameters
    ----------
    worker_name : str
        Name of the worker
    source : str, optional
        Source component name
        
    Returns
    -------
    BridgicLogger
        BridgicLogger instance for worker logging
    """
    return BridgicLogger(f"{WORKER_LOGGER_NAME}.{worker_name}", source)


def get_automa_logger(automa_name: str, source: Optional[str] = None) -> BridgicLogger:
    """Get an automa logger instance.
    
    Parameters
    ----------
    automa_name : str
        Name of the automa
    source : str, optional
        Source component name
        
    Returns
    -------
    BridgicLogger
        BridgicLogger instance for automa logging
    """
    return BridgicLogger(f"{AUTOMA_LOGGER_NAME}.{automa_name}", source)


def get_model_logger(model_name: str, source: Optional[str] = None) -> BridgicLogger:
    """Get a model logger instance.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    source : str, optional
        Source component name
        
    Returns
    -------
    BridgicLogger
        BridgicLogger instance for model logging
    """
    return BridgicLogger(f"{MODEL_LOGGER_NAME}.{model_name}", source)


def get_event_logger(source: Optional[str] = None) -> BridgicLogger:
    """Get an event logger instance.
    
    Parameters
    ----------
    source : str, optional
        Source component name
        
    Returns
    -------
    BridgicLogger
        BridgicLogger instance for event logging
    """
    return BridgicLogger(EVENT_LOGGER_NAME, source)


def get_trace_logger(source: Optional[str] = None) -> BridgicLogger:
    """Get a trace logger instance.
    
    Parameters
    ----------
    source : str, optional
        Source component name
        
    Returns
    -------
    BridgicLogger
        BridgicLogger instance for trace logging
    """
    return BridgicLogger(TRACE_LOGGER_NAME, source)


def setup_logging(
    level: int = logging.INFO,
    enable_trace: bool = True,
    enable_events: bool = True,
    enable_workers: bool = True,
    enable_automa: bool = True,
    enable_model: bool = True,
    handlers: Optional[list] = None,
    component_levels: Optional[Dict[str, int]] = None,
) -> None:
    """Setup logging configuration for Bridgic.
    
    Parameters
    ----------
    level : int, default=logging.INFO
        Default logging level for all components
    enable_trace : bool, default=True
        Whether to enable trace logging (DEBUG level)
    enable_events : bool, default=True
        Whether to enable structured event logging (DEBUG level)
    enable_workers : bool, default=True
        Whether to enable worker logging (INFO level)
    enable_automa : bool, default=True
        Whether to enable automa logging (INFO level)
    enable_model : bool, default=True
        Whether to enable model logging (INFO level)
    handlers : list, optional
        Custom log handlers to add to the root logger
    component_levels : dict, optional
        Custom logging levels for specific components.
        Keys should be component names (e.g., 'workers', 'automa', 'model', 'events', 'trace')
        and values should be logging level constants.
    """
    # Configure root logger
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(level)
    
    if handlers:
        for handler in handlers:
            root_logger.addHandler(handler)
    else:
        # Default console handler with improved formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Configure component-specific loggers
    component_configs = {
        'trace': (TRACE_LOGGER_NAME, logging.DEBUG if enable_trace else logging.CRITICAL),
        'events': (EVENT_LOGGER_NAME, logging.DEBUG if enable_events else logging.CRITICAL),
        'workers': (WORKER_LOGGER_NAME, logging.INFO if enable_workers else logging.CRITICAL),
        'automa': (AUTOMA_LOGGER_NAME, logging.INFO if enable_automa else logging.CRITICAL),
        'model': (MODEL_LOGGER_NAME, logging.INFO if enable_model else logging.CRITICAL),
    }
    
    # Apply custom component levels if provided
    if component_levels:
        for component, custom_level in component_levels.items():
            if component in component_configs:
                component_configs[component] = (component_configs[component][0], custom_level)
    
    # Configure each component logger
    for component, (logger_name, component_level) in component_configs.items():
        component_logger = logging.getLogger(logger_name)
        component_logger.setLevel(component_level)
        # Ensure the component logger inherits handlers from root logger
        component_logger.propagate = True


def setup_development_logging() -> None:
    """Setup logging configuration optimized for development.
    
    This configuration enables all logging levels and components,
    making it easy to debug and develop with the Bridgic framework.
    """
    setup_logging(
        level=logging.DEBUG,
        enable_trace=True,
        enable_events=True,
        enable_workers=True,
        enable_automa=True,
        enable_model=True,
    )


def setup_production_logging() -> None:
    """Setup logging configuration optimized for production.
    
    This configuration disables verbose logging while keeping
    important operational information visible.
    """
    setup_logging(
        level=logging.WARNING,
        enable_trace=False,
        enable_events=False,
        enable_workers=True,
        enable_automa=True,
        enable_model=True,
    )


def setup_quiet_logging() -> None:
    """Setup minimal logging configuration.
    
    This configuration only shows critical errors and warnings.
    """
    setup_logging(
        level=logging.ERROR,
        enable_trace=False,
        enable_events=False,
        enable_workers=False,
        enable_automa=False,
        enable_model=False,
    )


def setup_verbose_logging() -> None:
    """Setup verbose logging configuration.
    
    This configuration shows all possible logging information,
    useful for detailed debugging and analysis.
    """
    setup_logging(
        level=logging.DEBUG,
        enable_trace=True,
        enable_events=True,
        enable_workers=True,
        enable_automa=True,
        enable_model=True,
        component_levels={
            'trace': logging.DEBUG,
            'events': logging.DEBUG,
            'workers': logging.DEBUG,
            'automa': logging.DEBUG,
            'model': logging.DEBUG,
        }
    )
