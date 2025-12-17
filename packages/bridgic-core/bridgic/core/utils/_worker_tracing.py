from typing import Any, Dict

from bridgic.core.automa import Automa
from bridgic.core.automa.worker import Worker

def get_worker_tracing_step_name(key: str, worker: Worker) -> str:
	"""
	Get the step name for a worker, including nested automa name if applicable.
	
	Parameters
	----------
	key : str
		The worker key identifier.
	worker : Worker
		The worker instance.
	
	Returns
	-------
	str
		The step name for the worker.
	"""
	from bridgic.core.automa._graph_automa import _GraphAdaptedWorker
	if isinstance(worker, _GraphAdaptedWorker) and worker.is_automa():
		nested_automa = worker._decorated_worker
		return f"{key}  <{nested_automa.name}>"
	return key

def build_worker_tracing_dict(worker: Worker, parent: "Automa") -> Dict[str, Any]:
	"""
	Build worker tracing information as a dictionary.
	
	Parameters
	----------
	worker : Worker
		The worker instance to build tracing information for.
	parent : Automa
		The parent automa instance containing this worker.
	
	Returns
	-------
	Dict[str, Any]
		A dictionary containing worker tracing information.
	"""
	report_info = worker.get_report_info()

	other_report_info = {
		"nesting_level": 0,
		"parent_automa_name": parent.name,
		"parent_automa_class": parent.__class__.__name__,
	}
	# Calculate nesting level
	current = parent
	nesting_level = 1 
	while True:
		if current.is_top_level():
			other_report_info["nesting_level"] = nesting_level
			break
		else:
			current = current.parent
			nesting_level += 1

	# Get top-level automa
	top = worker._get_top_level_automa()
	other_report_info["top_automa_name"] = top.name

	return {
		**other_report_info,
		**report_info,
	}
