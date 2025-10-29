import inspect
from typing import Callable, Dict, Any, List, Optional, cast, get_origin, Annotated
from pydantic import create_model, WithJsonSchema, Field, ConfigDict
from pydantic.fields import FieldInfo
from bridgic.core.automa.args._args_descriptor import JSON_SCHEMA_IGNORE_ARG_TYPES
from docstring_parser import parse as parse_docstring # type: ignore

def create_func_params_json_schema(
    func: Callable,
    ignore_params: List[str] = ["self", "cls"],
) -> Dict[str, Any]:
    """
    Create a JSON schema for the parameters of a function. The docstring of the function will be used to extract the description of the parameters, if available.

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
    def _get_param_description_from_annotation(annotated_type: Any) -> Optional[str]:
        for metadata in annotated_type.__metadata__:
            if isinstance(metadata, FieldInfo):
                return metadata.description
            if isinstance(metadata, WithJsonSchema) and metadata.json_schema:
                return metadata.json_schema.get("description")
        return None

    sig = inspect.signature(func)
    docstring = parse_docstring(func.__doc__)
    field_defs: Dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in ignore_params:
            continue

        # First: Resolve the type of the parameter
        param_type = param.annotation
        if param_type is inspect.Parameter.empty:
            # In the case of no annotation
            param_type = Any

        # Second: Resolve the default of the parameter
        param_default = param.default
        if isinstance(param_default, JSON_SCHEMA_IGNORE_ARG_TYPES):
            continue
        # param_default may be inspect.Parameter.empty in the case of no default.
    
        # Third: Resolve the description of the parameter
        # The description of a parameter may come from multiple sources, with the following order of priority (from highest to lowest):
        # 1. param: type = Field(description="description here", ...)
        # 2. param: Annotated[type, "description here"]
        # 3. param: Annotated[type, Field(description="description here", ...)]
        # 4. param: Annotated[type, WithJsonSchema({"description":"description here", ...})]
        # 5. the params section of the function's docstring
        param_description = None
        need_overwritten = False # Whether the description of the parameter needs to be overwritten.
        if isinstance(param_default, FieldInfo):
            param_description = param_default.description
        if not param_description and get_origin(param_type) is Annotated:
            if len(param_type.__metadata__) == 1 and isinstance(param_type.__metadata__[0], str):
                param_description = cast(str, param_type.__metadata__[0])
                need_overwritten = bool(param_description)
            else:
                param_description = _get_param_description_from_annotation(param_type)
        if not param_description and docstring.params:
            # Search the param description in the docstring.params
            for p in docstring.params:
                if p.arg_name == param_name:
                    param_description = p.description
                    need_overwritten = bool(param_description)
                    break
        
        # Assemble the Field object and add it to the field_defs
        if need_overwritten:
            if param_default is inspect.Parameter.empty:
                field = Field(description=param_description)
            elif isinstance(param_default, FieldInfo):
                field = param_default
                field.description = param_description
            else:
                field = Field(default=param_default, description=param_description)
            field_defs[param_name] = (param_type, field)
        else:
            if param_default is inspect.Parameter.empty:
                field_defs[param_name] = param_type
            else:
                field_defs[param_name] = (param_type, param_default)
    
    # Note: Set arbitrary_types_allowed=True to allow custom parameter types that support JSON schema, by implementting `__get_pydantic_core_schema__` or `__get_pydantic_json_schema__`.
    JsonSchemaModel = create_model(
        func.__name__,
        __config__=ConfigDict(arbitrary_types_allowed=True), 
        **field_defs
    )
    return JsonSchemaModel.model_json_schema()