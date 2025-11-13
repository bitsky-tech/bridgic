from typing import Dict

from bridgic.core.automa._graph_automa import _GraphAdaptedWorker
from bridgic.core.automa.worker import Worker
from bridgic.core.types._worker_types import WorkerExecutionContext

def get_worker_exec_context(worker: "_GraphAdaptedWorker") -> WorkerExecutionContext:
	"""
	Build execution context information for a worker for logging and tracing.
	"""
	report_info = {}
	parent = getattr(worker, "parent", None)
	if parent is not None and hasattr(parent, "_workers"):
		# Search for this worker instance in parent's workers dict
		# Note: In GraphAutoma, workers might be wrapped in _GraphAdaptedWorker
		workers: Dict[str, Worker] = parent._workers
		for wkey, worker_obj in workers.items():
			# Check if it's directly this worker
			if worker_obj is worker:
				report_info = worker_obj.get_report_info()
				break
			# Check if it's a wrapped worker (_GraphAdaptedWorker case)
			elif hasattr(worker_obj, "_decorated_worker") and worker_obj._decorated_worker is worker:
				report_info = worker_obj.get_report_info()
				break

	other_report_info = {
		"nesting_level": 0,
	}
	# Find parent automa
	if parent is not None:
		other_report_info["parent_automa_name"] = getattr(parent, "name", None)
		other_report_info["parent_automa_class"] = parent.__class__.__name__

	# Calculate nesting level
	current = parent
	while current is not None:
		other_report_info["nesting_level"] += 1
		current = getattr(current, "parent", None)

	# Get top-level automa
	top = worker._get_top_level_automa()
	if top is not None:
		other_report_info["top_automa_name"] = top.name

	return WorkerExecutionContext(
		**other_report_info,
		**report_info,
	)


