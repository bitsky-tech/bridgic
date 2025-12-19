import inspect
from typing import Callable, Dict, Any, List, Optional, cast, get_origin, Annotated
from pydantic import create_model, WithJsonSchema, Field, ConfigDict
from pydantic.fields import FieldInfo
from bridgic.core.automa.args._args_descriptor import JSON_SCHEMA_IGNORE_ARG_TYPES
from docstring_parser import parse as parse_docstring # type: ignore

def _extract_description_from_annotated(annotated_type: Any) -> Optional[str]:
    for metadata in annotated_type.__metadata__:
        if isinstance(metadata, FieldInfo):
            return metadata.description
        if isinstance(metadata, WithJsonSchema) and metadata.json_schema:
            return metadata.json_schema.get("description")
    return None

def _resolve_param_description(
    param_name: str,
    param_type: Any,
    param_default: Any,
    docstring_params: Optional[List[Any]],
) -> Optional[str]:
    # Priority 1: Check if param_default is a FieldInfo and has a description.
    if isinstance(param_default, FieldInfo) and param_default.description:
        return param_default.description

    # Priority 2: Check for description inside Annotated.
    if get_origin(param_type) is Annotated:
        # Check for simple string description: Annotated[type, "description"].
        if len(param_type.__metadata__) == 1 and isinstance(param_type.__metadata__[0], str):
            return cast(str, param_type.__metadata__[0])
        # Check for FieldInfo or WithJsonSchema description inside Annotated.
        description = _extract_description_from_annotated(param_type)
        if description:
            return description

    # Priority 3: Look for description from docstring
    if docstring_params:
        for p in docstring_params:
            if p.arg_name == param_name:
                return p.description

    return None

def _build_field_definition(
    param_type: Any,
    param_default: Any,
    description: Optional[str],
) -> Any:
    has_default = False
    actual_default = None

    if isinstance(param_default, FieldInfo):
        # If param_default is FieldInfo, check its default attribute.
        if param_default.default is not inspect.Parameter.empty:
            has_default = True
            actual_default = param_default.default
    elif param_default is not inspect.Parameter.empty:
        # If param_default is a raw value (could even be None), just use it.
        has_default = True
        actual_default = param_default

    if description is None:
        if not has_default:
            return param_type
        elif isinstance(param_default, FieldInfo):
            return (param_type, param_default)
        else:
            return (param_type, actual_default)

    field_kwargs = {"description": description}

    if has_default:
        field_kwargs["default"] = actual_default

    if isinstance(param_default, FieldInfo):
        if hasattr(param_default, "alias") and param_default.alias:
            field_kwargs["alias"] = param_default.alias
        if hasattr(param_default, "title") and param_default.title:
            field_kwargs["title"] = param_default.title

    return (param_type, Field(**field_kwargs))

def create_func_params_json_schema(
    func: Callable,
    ignore_params: List[str] = ["self", "cls"],
) -> Dict[str, Any]:
    """
    Generate a JSON schema representing the input parameters of a callable. If the function's 
    docstring provides parameter descriptions, they will be incorporated into the schema. 
    Otherwise, the function's signature will be used to generate the schema.

    Parameters
    ----------
    func : Callable
        The function to create a JSON schema for.
    ignore_params : List[str]
        The parameters to ignore, not included in the resulting JSON schema.

    Returns
    -------
    Dict[str, Any]
        The JSON schema for the parameters of the function.
    """
    sig = inspect.signature(func)
    docstring = parse_docstring(func.__doc__)
    field_defs: Dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in ignore_params:
            continue

        param_type = param.annotation
        if param_type is inspect.Parameter.empty:
            param_type = Any

        param_default = param.default
        if isinstance(param_default, JSON_SCHEMA_IGNORE_ARG_TYPES):
            continue

        # Resolve parameter description in order of priority.
        param_description = _resolve_param_description(
            param_name,
            param_type,
            param_default,
            docstring.params if docstring else None,
        )

        # Build the field definition to the current parameter.
        field_def = _build_field_definition(param_type, param_default, param_description)
        field_defs[param_name] = field_def
    
    # Note: Set arbitrary_types_allowed=True to allow custom parameter types that support JSON schema, 
    # by implementting `__get_pydantic_core_schema__` or `__get_pydantic_json_schema__`.
    JsonSchemaModel = create_model(
        func.__name__,
        __config__=ConfigDict(arbitrary_types_allowed=True), 
        **field_defs
    )
    return JsonSchemaModel.model_json_schema()