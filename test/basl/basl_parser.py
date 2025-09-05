import re
from typing import Any

from lark import Lark


BASL = r"""
    ?start: automa

    # automa 表达式
    automa: graph_automa | sequential_automa | concurrent_automa
    graph_automa: graph_start automa_content graph_end
    sequential_automa: sequential_start automa_content sequential_end
    concurrent_automa: concurrent_start automa_content concurrent_end
    graph_end: CLOSE_OPEN GRAPH RANGLE WS*
    sequential_end: CLOSE_OPEN SEQUENTIAL RANGLE WS*
    concurrent_end: CLOSE_OPEN CONCURRENT RANGLE WS*
    graph_start: WS* LANGLE GRAPH exprs RANGLE
    sequential_start: WS* LANGLE SEQUENTIAL exprs RANGLE
    concurrent_start: WS* LANGLE CONCURRENT exprs RANGLE
    automa_content: (python_expr | concurrent_automa | graph_automa | sequential_automa | worker)+
    GRAPH: "graph"
    SEQUENTIAL: "sequential"
    CONCURRENT: "concurrent"
    python_expr: WS* PYTHONEXPRTOKEN WS*

    # workers 表达式
    worker: WS* LANGLE tag exprs SELF_CLOSE WS*

    # 属性表达式
    exprs: expr*
    expr: WS* tag WS* "=" WS* item WS*
    item: list | value | PYTHONEXPRTOKEN

    # list 列表
    list: LBRACKET [element ("," element)*] RBRACKET
    element: WS* element_content WS*
    element_content: ESCAPED_STRING | INNER_CONTENT | PYTHONEXPRTOKEN

    # 标识符和值
    tag: IDENTIFIER
    value: IDENTIFIER | ESCAPED_STRING

    # 基本符号
    PYTHONEXPRTOKEN: "PYTHONEXPRTOKEN"
    DB_LBRACE: "{{"
    DB_RBRACE: "}}"
    LBRACE: "{"
    RBRACE: "}"
    LBRACKET: "["
    RBRACKET: "]"
    SELF_CLOSE: "/>"
    CLOSE_OPEN: "</"
    LANGLE: "<"
    RANGLE: ">"
    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    INNER_CONTENT: /[^{}<>\[\]\/\s]+/  # 排除花括号、尖括号、斜杠、空白

    %import common.WS
    %import common.ESCAPED_STRING
    %ignore WS
"""


class BASLparser:
    def __init__(self):
        self.python_expr_token = None
        self.ast = None

    def _pre_process(self, input_code: str) -> str:
        """
        Preprocess the input BASL language:
            1. replace the {{ ... }} code block with specitial token: PYTHONEXPRTOKEN
            2. record the position of the {{ ... }} code block
        
        Args:
            input_code: the input BASL code
        
        Returns:
            the input BASL code after pre-process
        """
        # use stack to get the position of the {{ ... }} code block
        match_expr = []
        stack = []
        i = 0
        while i < len(input_code):
            if input_code.startswith("{{", i):
                stack.append(i)
                i += 2
                continue
            if input_code.startswith("}}}", i):
                i += 1
                continue
            if input_code.startswith("}}", i):
                if stack:
                    st = stack.pop()
                    ed = i + 2
                    match_expr.append((st, ed, input_code[st:ed], []))
                else:
                    # TODO: 抛出 {{ }} 不匹配错误，有多余的 }} 在位置 i 处
                    pass
                i += 2
                continue
            i += 1
        if stack:
            # TODO: 抛出 {{ }} 不匹配错误，有多余的 {{ 在位置 stack.pop() 处
            pass

        # according to the block start and end position, and nest relationship, build hierarchical tree
        sorted_match_expr = sorted(match_expr, key=lambda x: x[0])
        root = [-1, float('inf'), '', []]
        match_nest_expr = [root]
        for idx, (st, ed, expr, _) in enumerate(sorted_match_expr):
            node = [st, ed, expr, []]
            while not (match_nest_expr[-1][0] < st and ed < match_nest_expr[-1][1]):
                match_nest_expr.pop()
            match_nest_expr[-1][3].append(node)

            if idx != len(sorted_match_expr) - 1:
                match_nest_expr.append(node)

        # according to the block start and end position, replace top level {{ ... }} code block with PYTHONEXPRTOKEN
        top_level_node = match_nest_expr[0][3]
        BASL_code = input_code
        for node in reversed(top_level_node):
            BASL_code = BASL_code[:node[0]] + "PYTHONEXPRTOKEN" + BASL_code[node[1]:]
        
        # after completing replacement, record the position of the PYTHONEXPRTOKEN
        python_expr_token_pos = []
        i = 0
        while i < len(BASL_code):
            if BASL_code.startswith("PYTHONEXPRTOKEN", i):
                python_expr_token_pos.append((i, i + len("PYTHONEXPRTOKEN")))
                i += len("PYTHONEXPRTOKEN")
            else:
                i += 1
        for node, pos in zip(top_level_node, python_expr_token_pos):
            node.append(pos)


        print(f'match_nest_expr: {json.dumps(match_nest_expr, indent=4, ensure_ascii=False)}')
        self.python_expr_token = match_nest_expr
        self.ast = self._parser(BASL_code)
        return BASL_code
                

    def _parser(self, input_after_pre_process: str) -> Any:
        """
        Parse the input BASL code:
            1. parse the input with Lark parser
            2. ...
        
        Args:
            input_after_pre_process: the input BASL code

        Returns:
            An AST tree of the input BASL code
        """
        lark_parser = Lark(BASL, start='start', parser='lalr')
        ast = lark_parser.parse(input_after_pre_process)
        return ast


####################################################################################################################################################
# 测试
####################################################################################################################################################

import json

if __name__ == "__main__":
    test_BASL = """<graph>
            <DivideGameProgrammingTask
                game_requirement={{user_input}}
                output_key="subtasks"
                is_start="true"
            />
            <sequential
                id="SequentialGenerateGameModule"
                bindle_mode="true"
                output_key="module_list"
                dependencies=["DivideGameProgrammingTask"]
            >
            {{
                [<ProgrammingGameModule 
                    user_requirement={{user_input}}
                    task={{t}} 
                    output_key={{f"game_module_{i}"}}
                    generated_modules={{
                        [module_list[j] for j in range(i)]
                    }}
                    /> 
                    for i,t in enumerate(subtasks)
                ]
            }}
            </sequential>
            <CombineWholeGame
                user_requirement={{user_input}}
                game_modules={{module_list}}
                is_output="true"
                dependencies=["SequentialGenerateGameModule"]
            />
      </graph>"""

    
    test_BASL_after_pre_process = """<graph>
            <DivideGameProgrammingTask
                game_requirement=PYTHONEXPRTOKEN
                output_key="subtasks"
                is_start="true"
            />
            <sequential
                id="SequentialGenerateGameModule"
                bindle_mode="true"
                output_key="module_list"
                dependencies=["DivideGameProgrammingTask"]
            >
            PYTHONEXPRTOKEN
            </sequential>
            <CombineWholeGame
                user_requirement=PYTHONEXPRTOKEN
                game_modules=PYTHONEXPRTOKEN
                is_output="true"
                dependencies=["SequentialGenerateGameModule"]
            />
      </graph>"""

    bridgic_parser = BASLparser()

    BASL_code = bridgic_parser._pre_process(test_BASL)
    print(f'BASL_code: {BASL_code}')

    bridgic_parser._parser(test_BASL_after_pre_process)



