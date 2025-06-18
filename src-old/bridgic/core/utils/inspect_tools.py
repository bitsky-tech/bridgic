import inspect

def get_first_arg_type(sig: inspect.Signature) -> type:
    for _, param in sig.parameters.items():
        first_arg_type = param.annotation
        break
    return first_arg_type
