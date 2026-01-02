
from threading import Lock
from typing import Dict, Union, Literal, Callable

color_funcs: Dict[str, Callable[[str], str]] = {
    "blue":   lambda msg: "\033[38;5;21m{}\033[0m".format(msg),
    "cyan":   lambda msg: "\033[38;5;51m{}\033[0m".format(msg),
    "green":  lambda msg: "\033[38;5;46m{}\033[0m".format(msg),
    "lime":   lambda msg: "\033[38;5;118m{}\033[0m".format(msg),
    "olive":  lambda msg: "\033[38;5;100m{}\033[0m".format(msg),
    "yellow": lambda msg: "\033[38;5;226m{}\033[0m".format(msg),
    "orange": lambda msg: "\033[38;5;208m{}\033[0m".format(msg),
    "brown":  lambda msg: "\033[38;5;130m{}\033[0m".format(msg),
    "red":    lambda msg: "\033[38;5;196m{}\033[0m".format(msg),
    "pink":   lambda msg: "\033[38;5;213m{}\033[0m".format(msg),
    "purple": lambda msg: "\033[38;5;129m{}\033[0m".format(msg),
    "gray":   lambda msg: "\033[38;5;244m{}\033[0m".format(msg),
    "white":  lambda msg: "\033[38;5;15m{}\033[0m".format(msg),
}

legal_colors = list(color_funcs.keys())

def colored(
    msg: str,
    color: Union[Literal["red", "green", "yellow", "blue", "purple", "cyan", "white", "gray", "orange", "pink", "brown", "lime", "olive"], None] = None,
) -> str:
    if color not in legal_colors:
        raise ValueError(f"Invalid color: {color}")
    return color_funcs[color](msg)

class Printer:
    """
    A thread-safe printer class that can print colored text to the console.

    Parameters
    ----------
    color : Union[Literal["red", "green", "yellow", "blue", "purple", "cyan", "white", "gray", "orange", "pink", "brown", "lime", "navy", "teal", "olive"], None]
        The color of the text. If None, the text will be printed in the default color.
    """
    lock: Lock = Lock()

    @classmethod
    def print(
        cls,
        *values,
        color: Union[Literal["red", "green", "yellow", "blue", "purple", "cyan", "white", "gray", "orange", "pink", "brown", "lime", "olive"], None] = None,
        **kwargs,
    ):
        if color:
            if color not in legal_colors:
                raise ValueError(f"Invalid color: {color}")
            values = list(map(lambda x: color_funcs[color](x), values))

        with cls.lock:
            print(*values, **kwargs, flush=True)

printer = Printer()