from typing import Tuple, Dict, Any, List
from inspect import Parameter, _ParameterKind


def safely_map_args(
        in_args: Tuple[Any, ...], 
        in_kwargs: Dict[str, Any],
        rx_param_names_dict: Dict[_ParameterKind, List[Tuple[str, Any]]],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        
        def get_param_names(param_names_dict: List[Tuple[str, Any]]) -> List[str]:
            return [name for name, _ in param_names_dict]
            
        positional_only_param_names = get_param_names(rx_param_names_dict.get(Parameter.POSITIONAL_ONLY, []))
        positional_or_keyword_param_names = get_param_names(rx_param_names_dict.get(Parameter.POSITIONAL_OR_KEYWORD, []))

        # Resolve the positional arguments `rx_args`.
        positional_param_names = positional_only_param_names + positional_or_keyword_param_names
        var_positional_param_names = get_param_names(rx_param_names_dict.get(Parameter.VAR_POSITIONAL, []))
        if len(in_args) == 1 and in_args[0] is None and len(positional_param_names) == 0 and len(var_positional_param_names) == 0:
            # The special case of the predecessor returning None and the successor has no arguments expected.
            rx_args = ()
        else:
            # In normal cases, positional arguments are unchanged.
            rx_args = in_args

        # Resolve the keyword arguments `rx_kwargs`.
        # keyword arguments are firstly filtered by `masked_keyword_param_names`.
        if len(in_args) > len(positional_only_param_names):
            masked_keyword_param_names = positional_or_keyword_param_names[:len(in_args)-len(positional_only_param_names)]
        else:
            masked_keyword_param_names = []
        rx_kwargs = {k:v for k,v in in_kwargs.items() if k not in masked_keyword_param_names}
        var_keyword_param_names = get_param_names(rx_param_names_dict.get(Parameter.VAR_KEYWORD, []))
        if not var_keyword_param_names:
            # keyword arguments are secondly filtered by `keyword_param_names`.
            keyword_only_param_names = get_param_names(rx_param_names_dict.get(Parameter.KEYWORD_ONLY, []))
            keyword_param_names = positional_or_keyword_param_names + keyword_only_param_names
            rx_kwargs = {k:v for k,v in rx_kwargs.items() if k in keyword_param_names}
        return rx_args, rx_kwargs