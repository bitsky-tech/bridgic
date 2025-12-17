
from threading import Lock
from typing import Dict, Union, Literal, Callable

legal_colors = ["red", "green", "yellow", "blue", "purple", "cyan", "white", "magenta"]

color_funcs: Dict[str, Callable[[str], str]] = {
    "red": lambda msg: "\033[91m{}\033[00m".format(msg),
    "green": lambda msg: "\033[92m{}\033[00m".format(msg), 
    "yellow": lambda msg: "\033[93m{}\033[00m".format(msg),
    "blue": lambda msg: "\033[94m{}\033[00m".format(msg),
    "purple": lambda msg: "\033[95m{}\033[00m".format(msg),
    "cyan": lambda msg: "\033[96m{}\033[00m".format(msg),
    "white": lambda msg: "\033[97m{}\033[00m".format(msg),
    "magenta": lambda msg: "\033[35m{}\033[00m".format(msg)
}

def colored(
    msg: str,
    color: Union[Literal["red", "green", "yellow", "blue", "purple", "cyan", "white", "magenta"], None] = None,
) -> str:
    if color not in legal_colors:
        raise ValueError(f"Invalid color: {color}")
    return color_funcs[color](msg)

class Printer:
    """
    A thread-safe printer class that can print colored text to the console.

    Parameters
    ----------
    color : Union[Literal["red", "green", "yellow", "blue", "purple", "cyan", "white", "magenta"], None]
        The color of the text. If None, the text will be printed in the default color.
    """
    lock: Lock = Lock()

    @classmethod
    def print(
        cls,
        *values,
        color: Union[Literal["red", "green", "yellow", "blue", "purple", "cyan", "white", "magenta"], None] = None,
        **kwargs,
    ):
        if color:
            if color not in legal_colors:
                raise ValueError(f"Invalid color: {color}")
            values = list(map(lambda x: color_funcs[color](x), values))

        with cls.lock:
            print(*values, **kwargs, flush=True)

printer = Printer()