# Re-export from new canonical location
# This module has been reorganized to victor.storage.embeddings/

# Re-export all public symbols
from victor.storage.embeddings import *

# Re-export submodules for backwards compatibility
from victor.storage.embeddings import service
from victor.storage.embeddings import collections
from victor.storage.embeddings import intent_classifier
from victor.storage.embeddings import task_classifier
