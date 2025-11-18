from typing import Any, Dict

from bridgic.core.automa import Automa
from bridgic.core.automa._graph_automa import _GraphAdaptedWorker

def get_worker_exec_context(worker: "_GraphAdaptedWorker", parent: "Automa") -> Dict[str, Any]:
	"""
	Build execution context information for a worker for logging and tracing.
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
