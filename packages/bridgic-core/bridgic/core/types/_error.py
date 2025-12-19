###########################################################
# Worker Errors
###########################################################

class WorkerSignatureError(Exception):
    """
    Raised when invalid signature is detected in the case of defining a worker.
    """
    pass

class WorkerArgsMappingError(Exception):
    """
    Raised when the parameters declaration of a worker does not meet the requirements of the arguments mapping rule.
    """
    pass

class WorkerArgsInjectionError(Exception):
    """
    Raised when the arguments injection mechanism encountered an error during operation.
    """
    pass

class WorkerRuntimeError(RuntimeError):
    """
    Raised when the worker encounters an unexpected error during runtime.
    """
    pass

###########################################################
# Automa Errors
###########################################################

class AutomaDeclarationError(Exception):
    """
    Raised when the declaration of workers within an Automa is not valid.
    """
    pass

class AutomaCompilationError(Exception):
    """
    Raised when the compilation or validation of an Automa fails.
    """
    pass

class AutomaRuntimeError(RuntimeError):
    """
    Raised when the execution of an Automa encounters an unexpected error.
    """
    pass

###########################################################
# Prompt Errors
###########################################################

class PromptSyntaxError(Exception):
    pass

class PromptRenderError(RuntimeError):
    pass