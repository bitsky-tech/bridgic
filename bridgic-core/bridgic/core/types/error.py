###########################################################
# Worker Errors
###########################################################

class WorkerSignatureError(Exception):
    pass

class WorkerArgsMappingError(Exception):
    pass

class WorkerRuntimeError(RuntimeError):
    pass

###########################################################
# Automa Errors
###########################################################

class AutomaDeclarationError(Exception):
    pass

class AutomaCompilationError(Exception):
    pass

class AutomaRuntimeError(RuntimeError):
    pass

class AutomaDataInjectionError(Exception):
    pass

###########################################################
# Prompt Errors
###########################################################

class PromptSyntaxError(Exception):
    pass

class PromptRenderError(RuntimeError):
    pass