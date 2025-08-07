class WorkerSignatureError(Exception):
    pass

class WorkerArgsMappingError(Exception):
    pass

class AutomaDeclarationError(Exception):
    pass

class AutomaCompilationError(Exception):
    pass

class AutomaRuntimeError(RuntimeError):
    pass
