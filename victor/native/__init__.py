# Re-export from new canonical location
# This module has been reorganized to victor.processing.native/
from victor.processing.native import *

# Export accelerator modules
from victor.native.accelerators import (
    AstProcessorAccelerator,
    get_ast_processor,
    EmbeddingOpsAccelerator,
    get_embedding_accelerator,
    RegexEngineAccelerator,
    get_regex_engine,
    SignatureAccelerator,
    get_signature_accelerator,
    FileOpsAccelerator,
    get_file_ops_accelerator,
)

__all__ = [
    "AstProcessorAccelerator",
    "get_ast_processor",
    "EmbeddingOpsAccelerator",
    "get_embedding_accelerator",
    "RegexEngineAccelerator",
    "get_regex_engine",
    "SignatureAccelerator",
    "get_signature_accelerator",
    "FileOpsAccelerator",
    "get_file_ops_accelerator",
]
