from lark import Lark, Tree
from typing import Tuple, List, TYPE_CHECKING

from bridgic.asl.types import ASLUnmatchedBraceError

if TYPE_CHECKING:
    from bridgic.asl.compiler.entry import EntryContext


# ASL grammar
ASL = r"""
    ?start: automa

    # automa expression
    automa: graph_automa | sequential_automa | concurrent_automa
    graph_automa: "<" "graph" exprs ">" automa_content "</" "graph" ">"
    sequential_automa: "<" "sequential" exprs ">" automa_content "</" "sequential" ">"
    concurrent_automa: "<" "concurrent" exprs ">" automa_content "</" "concurrent" ">"
    automa_content: (python_expr | concurrent_automa | graph_automa | sequential_automa | worker)+

    # workers expression
    worker: "<" TAG exprs "/>"

    # attributes expression
    exprs: expr*
    expr: TAG "=" item
    item: list | VALUE | python_expr

    # list expression
    list: "[" [VALUE ("," VALUE)*] "]"

    # python expression block
    python_expr: "PYTHONEXPRTOKEN"

    # basic symbols
    TAG: /[a-zA-Z_][a-zA-Z0-9_]*/
    VALUE: ESCAPED_STRING

    %import common.WS
    %import common.ESCAPED_STRING
    %ignore WS
"""


class ASLparser:
    """
    Parser for ASL (Agent Structure Language) code.
    
    This class handles the parsing of ASL code into an Abstract Syntax Tree (AST).
    It performs preprocessing to handle embedded Python expressions and uses
    the Lark parser to generate the final AST representation.
    
    The parsing process involves several key steps:
    1. **Preprocessing**: Extract and replace Python expressions with tokens
    2. **AST Generation**: Use Lark parser to generate the final AST
    3. **Import Analysis**: Analyze worker imports and update context
    
    Attributes
    ----------
    context : EntryContext
        The compilation context that manages symbol tables and import information.
    input_code : str
        The original ASL input code before preprocessing.
    ASL_code : str
        The preprocessed ASL code with Python expressions replaced by tokens.
    python_expr_token : List
        Hierarchical structure containing Python expression information.
    ast : Tree
        The generated Abstract Syntax Tree representation.
    
    Main Methods
    -------
    parse(input_code)
        Main parsing method that processes ASL code and returns AST.
    
    Notes
    -----
    The parser handles nested Python expressions using a stack-based approach
    and builds a hierarchical tree structure to maintain proper nesting
    relationships during token replacement.
    """
    
    def __init__(self, context: "EntryContext"):
        """
        Initialize the ASL parser with compilation context.
        
        Parameters
        ----------
        context : EntryContext
            The compilation context that provides symbol table and import
            information for the parsing process.
        """
        self.context = context
        self.input_code = None
        self.ASL_code = None
        self.python_expr_token = None
        self.ast = None

    def parse(self, input_code: str) -> Tuple[Tree, List]:
        """
        Parse ASL code and return AST with Python expression tokens.
        
        This method performs the complete parsing process:
        1. Extracts Python expressions ({{ ... }}) from the input
        2. Builds hierarchical nesting structure
        3. Replaces top-level expressions with special tokens
        4. Generates AST using Lark parser
        5. Analyzes worker imports and updates context
        
        Parameters
        ----------
        input_code : str
            The ASL code string to parse, which may contain embedded
            Python expressions in {{ ... }} blocks.
        
        Returns
        -------
        Tuple[Tree, List]
            A tuple containing:
            - Tree: The parsed AST representation of the ASL code
            - List: Hierarchical structure containing Python expression
              information and positions
        
        Raises
        ------
        ASLUnmatchedBraceError
            If there are unmatched {{ or }} braces in the input code.
        
        Notes
        -----
        The method uses a stack-based algorithm to handle nested Python
        expressions and builds a hierarchical tree to maintain proper
        nesting relationships. Only top-level Python expressions are
        replaced with tokens to avoid conflicts with the grammar.
        """
        # Extract Python expressions using stack-based matching
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
                    raise ASLUnmatchedBraceError("}}", i)
                i += 2
                continue
            i += 1
        if stack:
            raise ASLUnmatchedBraceError("{{", stack.pop())

        # Build hierarchical tree structure based on nesting relationships
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

        # Replace top-level Python expressions with special tokens
        top_level_node = match_nest_expr[0][3]
        ASL_code = input_code
        for node in reversed(top_level_node):
            ASL_code = ASL_code[:node[0]] + "PYTHONEXPRTOKEN" + ASL_code[node[1]:]
        
        # Record positions of replaced tokens
        python_expr_token_pos = []
        i = 0
        while i < len(ASL_code):
            if ASL_code.startswith("PYTHONEXPRTOKEN", i):
                python_expr_token_pos.append((i, i + len("PYTHONEXPRTOKEN")))
                i += len("PYTHONEXPRTOKEN")
            else:
                i += 1
        for node, pos in zip(top_level_node, python_expr_token_pos):
            node.append(pos)

        # Store parsing results and generate AST
        self.input_code = input_code
        self.ASL_code = ASL_code
        self.python_expr_token = match_nest_expr
        self.ast = self._parse(ASL_code)

        # TODO: Flatten the nested python_expr_token structure for easier iteration
        return self.ast, self.python_expr_token
                

    def _parse(self, input_after_pre_process: str) -> Tree:
        """
        This internal method uses the Lark parser with the ASL grammar
        to generate the Abstract Syntax Tree from the preprocessed code.
        """
        lark_parser = Lark(ASL, start='start', parser='lalr')
        ast = lark_parser.parse(input_after_pre_process)
        self._parse_import_worker(ast)
        return ast

    def _parse_import_worker(self, tree: Tree):
        """
        Analyze worker imports and update compilation context using context.post_init_context().
        
        This method extracts worker names from the AST and matches them
        with import statements to determine their types and update the
        compilation context accordingly.
        """
        workers = [worker.children[0].value for worker in tree.find_data("worker")]
        import_models = self.context.get_import_list()

        import_code_list = []
        for worker in workers:
            for import_model in import_models:
                if worker in import_model["import_name"] and import_model["import_code"] not in import_code_list:
                    import_code_list.append(import_model["import_code"])
                    break
        if import_code_list:
            import_code = "\n".join(import_code_list)
            self.context.post_init_context(import_code, workers)
            

