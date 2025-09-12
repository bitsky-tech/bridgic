import inspect
import ast
from functools import wraps
from lark import Tree
from typing import Callable, Tuple, List, Dict


from bridgic.asl.compiler.parser import ASLparser
from bridgic.asl.compiler.translator import GraphAutomaTranslator
from bridgic.asl.types import PythonCompilationError, ASLCompilationError


class EntryContext:
    """
    Context manager for ASL entry compilation.
    
    This class manages the compilation context for ASL (Agent Structure Language)
    entries, including symbol table management, import analysis, and source code
    inspection. It provides the necessary context information for translating
    ASL code to Python code.
    
    Attributes
    ----------
    symbol_mapping : Dict[str, str]
        Mapping of symbol names to their types (e.g., 'function', 'class').
        Contains both locally defined symbols and imported symbols.
    import_list : List[Dict[str, Union[str, List[str]]]]
        List of import statements found in the source code. Each item contains
        'import_code' (the full import statement) and 'import_name' (list of
        imported names).
    entry_code : str
        The source code of the decorated function that contains ASL code.
    context_code : str
        The complete source code of the file containing the decorated function.
    args : List[Any]
        Positional arguments passed to the decorated function.
    kwargs : Dict[str, Any]
        Keyword arguments passed to the decorated function.
    
    Main Methods
    -------
    pre_init_context(func, args, kwargs)
        Initialize the context at the beginning of the compilation process with function information and arguments.
    post_init_context(import_code, workers)
        Initialize the context at the process of the generating ASL AST tree process with import statements and workers.
    get_symbol(symbol_name)
        Get the type of a symbol from the symbol mapping.
    """
    
    def __init__(self):
        self.symbol_mapping: Dict = {}
        self.import_list: List = []
        self.entry_code: str = ''
        self.context_code: str = ''
        self.args: List = []
        self.kwargs: Dict = {}

    def pre_init_context(self, func: Callable, args: List, kwargs: Dict):
        """
        This method performs the initial setup of the context by:
        1. Extracting source code from the decorated function
        2. Parsing the context code to build symbol mapping and import list
        3. Storing function arguments for later use
        
        Parameters
        ----------
        func : Callable
            The decorated function that contains ASL code.
        args : List[Any]
            Positional arguments passed to the decorated function.
        kwargs : Dict[str, Any]
            Keyword arguments passed to the decorated function.
        
        Notes
        -----
        This method should be called before any compilation operations.
        It sets up the foundation for symbol resolution and code generation.
        """
        self.entry_code, self.context_code = self._inspect_get_source_code(func)
        self.symbol_mapping, self.import_list = self._parse_context_code()
        self.args = args
        self.kwargs = kwargs

    def post_init_context(self, import_code: str, workers: List):
        """
        This method analyzes imported symbols to determine their types
        (function, class, etc.) and updates the symbol mapping accordingly.
        
        Parameters
        ----------
        import_code : str
            The import statement code to execute for type analysis.
        workers : List[str]
            List of worker names to analyze for type determination.
        
        Notes
        -----
        This method executes the import code in a controlled environment
        to safely determine the types of imported symbols without
        side effects on the main execution environment.
        """
        module_globals = {}
        exec(import_code, module_globals)

        symbol_mapping = {}
        for worker in workers:
            obj = module_globals.get(worker)
            if inspect.isfunction(obj):
                symbol_mapping[worker] = "function"
            elif inspect.isclass(obj):
                symbol_mapping[worker] = "class"
        
        self.symbol_mapping.update(symbol_mapping)
        
    def _inspect_get_source_code(self, func: Callable) -> Tuple[str, str]:
        """
        This method extracts both the specific function code and the complete
        file code containing the function.
        """
        source_file = inspect.getsourcefile(func)
        with open(source_file, 'r', encoding='utf-8') as f:
            context_code = f.read()
        entry_code = inspect.getsource(func)
        return entry_code, context_code

    def _parse_context_code(self) -> Tuple[Dict, List]:
        """
        This method uses AST parsing to analyze the source code and extract:
        1. Function and class definitions (for symbol mapping)
        2. Import statements (for import list)
        
        Returns
        -------
        Tuple[Dict[str, str], List[Dict[str, Union[str, List[str]]]]]
            A tuple containing:
            - symbol_mapping: Dictionary mapping symbol names to types
            - import_list: List of import statement dictionaries
        
        Raises
        ------
        PythonCompilationError
            If the context code cannot be parsed as valid Python code.
        
        Notes
        -----
        The method walks through the AST tree to identify different types
        of nodes and extracts relevant information for compilation context.
        """
        try:
            ast_tree = ast.parse(self.context_code)
            
            symbol_mapping = {}
            import_list = []
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    symbol_mapping[node.name] = "function"
                elif isinstance(node, ast.ClassDef):
                    symbol_mapping[node.name] = "class"
                elif isinstance(node, ast.Import):
                    import_modules = [node.names[i].name for i in range(len(node.names))]
                    import_list.append({
                        "import_code": f"import {', '.join(import_modules)}",
                        "import_name": import_modules,
                    })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    import_modules = [node.names[i].name for i in range(len(node.names))]
                    import_list.append({
                        "import_code": f"from {module} import {', '.join(import_modules)}",
                        "import_name": import_modules,
                    })
            return symbol_mapping, import_list
        except Exception as e:
            raise PythonCompilationError(e)
    
    def get_symbol(self, symbol_name: str) -> str:
        """
        Get the type of a symbol from the symbol mapping.
        
        Parameters
        ----------
        symbol_name : str
            The name of the symbol to look up.
        
        Returns
        -------
        str
            The type of the symbol ('function', 'class', etc.) or 'unknown'
            if the symbol is not found in the mapping.
        """
        symbol_info = self.symbol_mapping.get(symbol_name)
        if symbol_info:
            return symbol_info
        return 'unknown'

    def get_entry_code(self) -> str:
        """
        Get the entry function source code.
        """
        return self.entry_code

    def get_context_code(self) -> str:
        """
        Get the complete file source code.
        """
        return self.context_code

    def get_import_list(self) -> List:
        """
        Get the list of import statements.
        """
        return self.import_list

    def get_symbol_mapping(self) -> Dict:
        """
        Get the complete symbol mapping.
        """
        return self.symbol_mapping


class ASLEntry:
    """
    Decorator for compiling ASL (Agent Structure Language) code to Python code.
    
    This class provides a decorator interface for compiling ASL code into
    executable Python code. It manages the complete compilation pipeline from
    ASL source code to Python execution, including parsing, translation, and
    execution in the appropriate context.
    
    The compilation process consists of three main stages:
    1. **Parsing**: Convert ASL code to an Abstract Syntax Tree (AST)
    2. **Translation**: Transform the AST into executable Python code
    3. **Execution**: Run the generated Python code in the function's context
    
    Attributes
    ----------
    context : EntryContext
        The compilation context that manages symbol tables, imports, and
        source code information for the compilation process.
    
    Main Methods
    -------
    __call__(func)
        Decorator method that wraps functions containing ASL code.
    compile(asl_code)
        Compile ASL code string to Python code string.
    
    Examples
    --------
    >>> @ASLEntry()
    ... def my_workflow(user_input: int):
    ...     return '''
    ...     <graph>
    ...         <worker1 user_input={{user_input}} output_key="result" is_start="true" />
    ...     </graph>
    ...     '''
    >>> 
    >>> result = my_workflow(42)
    >>> print(result)  # Executes the compiled workflow
    
    Notes
    -----
    The decorator automatically handles:
    - Context initialization and symbol resolution
    - ASL to Python compilation
    - Code execution in the function's global namespace
    - Result extraction and return
    """
    
    def __init__(self):
        self.context = EntryContext()
    
    def __call__(self, func: Callable):
        """
        Decorator method that wraps functions containing ASL code.
        
        This method creates a wrapper function that:
        1. Initializes the compilation context
        2. Extracts ASL code from the decorated function
        3. Compiles ASL code to Python code
        4. Executes the generated Python code
        5. Returns the execution result
        
        Parameters
        ----------
        func : Callable
            The function to be decorated. This function should return
            a string containing ASL code.
        
        Returns
        -------
        Callable
            A wrapper function that handles ASL compilation and execution.
            The wrapper accepts the same arguments as the original function
            and returns the result of executing the compiled ASL code.
        
        Notes
        -----
        The wrapper function executes the generated Python code in the
        decorated function's global namespace, allowing access to all
        imported modules, functions, and classes defined in the same file.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize compilation context with function information
            self.context.pre_init_context(func, args, kwargs)

            # Extract ASL code from the decorated function
            asl_code = func(*args, **kwargs)
            
            # Compile ASL code to Python code
            asl_python_code = self.compile(asl_code)

            # Prepare execution environment using function's global namespace
            # TODO: 需要优化实际执行函数的传入机制，现在传入直接以 args 和 kargs 传入，不够清晰
            asl_globals = func.__globals__.copy()
            asl_globals.update(kwargs)
            for i, arg in enumerate(args):
                asl_globals[f'arg_{i}'] = arg
            
            # Execute the compiled Python code
            exec(asl_python_code, asl_globals)
            
            # Extract and return the result
            result = asl_globals.get('res', None)
            return result
        return wrapper
        
    def _parse(self, code: str) -> Tuple[Tree, List]:
        """
        Parse ASL code to Abstract Syntax Tree and extract Python expressions.
        
        This method uses the ASL parser to convert the input ASL code string
        into a structured AST representation, while also extracting any
        embedded Python expressions for later processing.
        
        Parameters
        ----------
        code : str
            The ASL code string to parse.
        
        Returns
        -------
        Tuple[Tree, List]
            A tuple containing:
            - Tree: The parsed AST representation of the ASL code
            - List: List of Python expression tokens found in the code
        
        Raises
        ------
        ASLCompilationError
            If the ASL code cannot be parsed.

        Notes
        -----
        The parser uses the compilation context to resolve symbols and
        validate the ASL syntax during parsing.
        """
        try:
            parser = ASLparser(self.context)
            return parser.parse(code)
        except Exception as e:
            raise ASLCompilationError(f"ASL code Syntax error: {e}")
    
    def _translate(self, ast: Tree, python_expr_token: List):
        """
        Translate AST to Python code using the GraphAutomaTranslator.
        
        This method takes the parsed AST and Python expression tokens
        and generates executable Python code that implements the ASL logic.
        
        Parameters
        ----------
        ast : Tree
            The parsed AST representation of the ASL code.
        python_expr_token : List
            List of Python expression tokens extracted during parsing.
        
        Returns
        -------
        str
            The generated Python code string that implements the ASL workflow.
        
        Notes
        -----
        The translator uses the compilation context to resolve symbol types
        and generate appropriate Python code for different worker types
        (functions, classes, etc.).
        """
        translator = GraphAutomaTranslator(python_expr_token, self.context)
        return translator.transform(ast)

    def compile(self, asl_code: str) -> str:
        """
        Compile ASL code string to Python code string.
        
        This is the main compilation method that orchestrates the complete
        compilation pipeline from ASL source code to executable Python code.
        
        Parameters
        ----------
        asl_code : str
            The ASL code string to compile.
        
        Returns
        -------
        str
            The compiled Python code string that can be executed to run
            the ASL workflow.
        
        Notes
        -----
        The compilation process involves:
        1. Parsing the ASL code to AST
        2. Translating the AST to Python code
        3. Returning the generated Python code
        
        This method is the main entry point for ASL compilation and is
        called automatically by the decorator wrapper.
        """
        ast, python_expr_token = self._parse(asl_code)
        code = self._translate(ast, python_expr_token)
        return code

