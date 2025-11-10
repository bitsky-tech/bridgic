import inspect
import time
import traceback
import uuid
from functools import wraps
from contextvars import ContextVar
from typing import Any, Callable, Dict

from bridgic.core.constants import STATUS_SUCCESS, STATUS_ERROR

# reference: https://docs.python.org/zh-cn/3.9/library/contextvars.html
_in_worker_log: ContextVar[bool] = ContextVar("in_worker_log", default=False)


def worker_logger(get_worker_name: Callable[[Any], str], method_label: str):
    """
    Decorator to emit structured worker start/end/error logs using BridgicLogger.

    Parameters
    ----------
    get_worker_name : Callable[[Any], str]
        Function that returns the worker name given `self`.
    method_label : str
        Method label such as 'arun' or 'run' written into metadata.
    """

    def decorator(func: Callable):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                # Guard: avoid nested/double logging within the same logical worker call
                token = None
                if _in_worker_log.get():
                    return await func(self, *args, **kwargs)
                token = _in_worker_log.set(True)
                execution_id = f"exec_{uuid.uuid4().hex[:16]}"
                context_info = self._get_execution_context()
                worker_key = context_info.worker_key
                dependencies = context_info.dependencies
                worker_name = get_worker_name(self)

                self._logger.log_worker_start(
                    worker_id=self._unique_id,
                    worker_name=worker_name,
                    input_data={"args": str(args), "kwargs": str(kwargs)},
                    execution_id=execution_id,
                    worker_key=worker_key,
                    dependencies=dependencies,
                    metadata={"method": method_label, **context_info.to_metadata_dict()},
                )

                start_time = time.time()
                try:
                    result = await func(self, *args, **kwargs)
                    self._logger.log_worker_end(
                        worker_id=self._unique_id,
                        worker_name=worker_name,
                        output_data={
                            "result_type": type(result).__name__,
                            "result": self._format_result_for_log(result),
                        },
                        execution_id=execution_id,
                        execution_time=time.time() - start_time,
                        worker_key=worker_key,
                        metadata={"method": method_label, "status": STATUS_SUCCESS, **context_info.to_metadata_dict()},
                    )
                    return result
                except Exception as e:
                    self._logger.log_worker_error(
                        worker_id=self._unique_id,
                        worker_name=worker_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        execution_id=execution_id,
                        error_traceback=traceback.format_exc(),
                        execution_time=time.time() - start_time,
                        worker_key=worker_key,
                        metadata={"method": method_label, "status": STATUS_ERROR, **context_info.to_metadata_dict()},
                    )
                    raise e
                finally:
                    if token is not None:
                        _in_worker_log.reset(token)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Guard: avoid nested/double logging within the same logical worker call
            token = None
            if _in_worker_log.get():
                return func(self, *args, **kwargs)
            token = _in_worker_log.set(True)
            execution_id = f"exec_{uuid.uuid4().hex[:16]}"
            context_info = self._get_execution_context()
            worker_key = context_info.worker_key
            dependencies = context_info.dependencies
            worker_name = get_worker_name(self)

            self._logger.log_worker_start(
                worker_id=self._unique_id,
                worker_name=worker_name,
                input_data={"args": str(args), "kwargs": str(kwargs)},
                execution_id=execution_id,
                worker_key=worker_key,
                dependencies=dependencies,
                metadata={"method": method_label, **context_info.to_metadata_dict()},
            )

            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                self._logger.log_worker_end(
                    worker_id=self._unique_id,
                    worker_name=worker_name,
                    output_data={
                        "result_type": type(result).__name__,
                        "result": self._format_result_for_log(result),
                    },
                    execution_id=execution_id,
                    execution_time=time.time() - start_time,
                    worker_key=worker_key,
                    metadata={"method": method_label, "status": STATUS_SUCCESS, **context_info.to_metadata_dict()},
                )
                return result
            except Exception as e:
                self._logger.log_worker_error(
                    worker_id=self._unique_id,
                    worker_name=worker_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    execution_id=execution_id,
                    error_traceback=traceback.format_exc(),
                    execution_time=time.time() - start_time,
                    worker_key=worker_key,
                    metadata={"method": method_label, "status": STATUS_ERROR, **context_info.to_metadata_dict()},
                )
                raise e
            finally:
                if token is not None:
                    _in_worker_log.reset(token)

        return sync_wrapper

    return decorator


