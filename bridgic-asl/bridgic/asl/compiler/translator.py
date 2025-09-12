from typing import List, TYPE_CHECKING
from enum import Enum
from lark import Transformer, Token, Tree

from bridgic.asl.utils import TokenIterator
from bridgic.asl.types import ASLWorkerNameNotFoundError
if TYPE_CHECKING:
    from bridgic.asl.compiler.entry import EntryContext

SPECIAL_TOKEN = 'PYTHONEXPRTOKEN'


class AutomaKeyWord(Enum):
    # TODO: 翻译到带属性的 Automa 时，考虑它的属性关键字
    pass


class WorkerKeyWord(Enum):
    KEY = "key"
    DEPENDENCIES = "dependencies"
    IS_START = "is_start"
    ARGS_MAPPING_RULE = "args_mapping_rule"
    IS_OUTPUT = "is_output"


class PythonExprParser:
    """
    A parser that parses the python expression token to python code.
    """
    def parse(self, python_expr_info: str):
        # TODO: 对表达式中，类似的 {{ ... }} 代码块，进行解析，甚至直接设计一个独立的文法解析翻译
        return python_expr_info


class CodeGenerator:
    pass


class GraphAutomaTranslator(Transformer):
    """
    A translator that translates the AST tree to python code.
    """
    KEY_WORDS_MAPPING = {
        "key": WorkerKeyWord.KEY,
        "dependencies": WorkerKeyWord.DEPENDENCIES,
        "is_start": WorkerKeyWord.IS_START,
        "args_mapping_rule": WorkerKeyWord.ARGS_MAPPING_RULE,
        "is_output": WorkerKeyWord.IS_OUTPUT,
    }

    def __init__(self, python_expr_token: List, context: "EntryContext"):
        super().__init__()
        self.python_expr_token_iter = TokenIterator(python_expr_token)
        self.context = context
        self.output_worker_key = None
    
    def resolve_python_expr_token(self):
        python_expr_info = next(self.python_expr_token_iter)
        return PythonExprParser().parse(python_expr_info)

    def automa(self, items):
        # TODO: 需要处理每一个 automa 来到这里时，代码的整合逻辑

        import_code = [
            "import asyncio",
            "from bridgic.core.automa.graph_automa import GraphAutoma",
            ""
        ]

        code_list = [
            *import_code,
            *items[0],
        ]

        gen_code = '\n'.join(code_list)
        return gen_code

    def graph_automa(self, items):
        automa_exprs = items[0]
        automa_content = items[1]
        
        # TODO: 处理 automa_exprs ，根据需要创建对应 automa

        local_args = ','.join([str(_) for _ in self.context.args])
        local_kwargs = ','.join([f'{k}={v}' for k,v in self.context.kwargs.items()])
        graph_automa_code = [
            "class GenAutoma(GraphAutoma): ...",
            f"automa_obj = GenAutoma(output_worker_key='{self.output_worker_key}')",
            *automa_content,
            f"res = asyncio.run(automa_obj.arun({local_args}, {local_kwargs}))",
            "print(res)"
        ]
        return graph_automa_code


    def sequential_automa(self, items):
        # TODO: 等待 sequential_automa 底层框架封装完成
        pass

    def concurrent_automa(self, items):
        # TODO: 等待 concurrent_automa 底层框架封装完成
        pass

    def automa_content(self, items):
        return items

    def worker(self, items):
        # TODO: 这里需要进一步细化处理，对 KEY 的判断
        # TODO: 这里的写法感觉怪怪的，有两次一样的判断，我想将这两个功能合成一个
        symbol_mapping = self.context.get_symbol_mapping()
        worker_tag = items[0].value
        worker_args = items[1]
        worker_type = symbol_mapping.get(worker_tag)

        args_code = []
        for arg in worker_args:
            tag, item = arg
            if self.KEY_WORDS_MAPPING.get(tag):
                if tag == "is_output":
                    self.output_worker_key = worker_tag
                else:
                    args_code.append(f'{tag}={item}')
        if "key" not in worker_args:  
            args_code.append(f'key="{worker_tag}"')

        if worker_type == "function":
            args_code.append(f'func={worker_tag}')
        elif worker_type == "class":
            args_code.append(f'worker_obj={worker_tag}()')
        else:
            raise ASLWorkerNameNotFoundError(worker_tag)
        args_code = ", ".join(args_code)

        worker_code = None        
        if worker_type == "function":
            worker_code = f"automa_obj.add_func_as_worker({args_code})"
        elif worker_type == "class":
            worker_code = f"automa_obj.add_worker({args_code})"
        else:
            raise ASLWorkerNameNotFoundError(worker_tag)

        return worker_code

    def exprs(self, items):
        return items

    def expr(self, items):
        tag = items[0].value
        item = items[1] if not isinstance(items[1], Token) else items[1].value
        return (tag, item)

    def item(self, items):
        return items[0]
    
    def list(self, items):
        list_data = []
        for item in items:
            if isinstance(item, Token):
                list_data.append(item.value.strip('"'))
            elif isinstance(item, Tree):
                # In current grammar, the list data type does not have elements with multi-symbol production rules, 
                # so there is no case where an item is a Tree. This might be added in the future.
                pass
        return list_data

    def python_expr(self, items):
        python_expr_code = self.resolve_python_expr_token()
        python_expr_value = python_expr_code[2].strip('{{').strip('}}')
        return python_expr_value

