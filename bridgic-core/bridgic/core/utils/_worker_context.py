from typing import Any, Dict, Optional

from bridgic.core.types._worker_types import WorkerExecutionContext


def get_worker_attributes(worker: Any) -> Dict[str, Any]:
	"""
	Get the attributes of a worker.
	"""
	if hasattr(worker, "key"):
		key = worker.key
	if hasattr(worker, "dependencies"):
		dependencies = worker.dependencies
	if hasattr(worker, "is_output"):
		is_output = worker.is_output
	if hasattr(worker, "is_start"):
		is_start = worker.is_start
	if hasattr(worker, "local_space"):
		local_space = worker.local_space
	return {
		"key": key,
		"dependencies": dependencies,
		"is_output": is_output,
		"is_start": is_start,
		"local_space": local_space,
	}

def get_worker_exec_context(worker: Any) -> WorkerExecutionContext:
	"""
	Build execution context information for a worker for logging and tracing.
	"""
	# Try to find worker_key from parent Automa
	key: Optional[str] = None
	dependencies: Optional[list] = None
	is_output: bool = False
	is_start: bool = False
	local_space: Optional[Dict[str, Any]] = None

	parent = getattr(worker, "parent", None)
	if parent is not None and hasattr(parent, "_workers"):
		# Search for this worker instance in parent's workers dict
		# Note: In GraphAutoma, workers might be wrapped in _GraphAdaptedWorker
		for wkey, worker_obj in parent._workers.items():
			# Check if it's directly this worker
			if worker_obj is worker:
				key = wkey
				attributes = get_worker_attributes(worker_obj)
				key = attributes["key"]
				dependencies = attributes["dependencies"]
				is_output = attributes["is_output"]
				is_start = attributes["is_start"]
				local_space = attributes["local_space"]
				break
			# Check if it's a wrapped worker (_GraphAdaptedWorker case)
			elif hasattr(worker_obj, "_decorated_worker") and worker_obj._decorated_worker is worker:
				key = wkey
				attributes = get_worker_attributes(worker_obj)
				key = attributes["key"]
				dependencies = attributes["dependencies"]
				is_output = attributes["is_output"]
				is_start = attributes["is_start"]
				local_space = attributes["local_space"]
				break

	# Find parent automa
	parent_automa_name: Optional[str] = None
	parent_automa_class: Optional[str] = None
	if parent is not None:
		parent_automa_name = getattr(parent, "name", None)
		parent_automa_class = parent.__class__.__name__

	# Calculate nesting level
	nesting_level = 0
	current = parent
	while current is not None:
		nesting_level += 1
		current = getattr(current, "parent", None)

	# Get top-level automa
	top_automa_name: Optional[str] = None
	top = worker
	while top is not None and getattr(top, "parent", None) is not None:
		top = top.parent
	if top is not None:
		top_automa_name = getattr(top, "name", None)

	return WorkerExecutionContext(
		key=key,
		dependencies=dependencies,
		parent_automa_name=parent_automa_name,
		parent_automa_class=parent_automa_class,
		nesting_level=nesting_level,
		top_automa_name=top_automa_name,
		is_output=is_output,
		is_start=is_start,
		local_space=local_space,
	)


