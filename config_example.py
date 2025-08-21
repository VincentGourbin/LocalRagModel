#!/usr/bin/env python3
"""
Configuration example for LocalRAG system
Copy to config.py and customize for your setup
"""

# =============================================================================
# STEP 01 - INDEXER CONFIGURATION
# =============================================================================

# Paths
DOCS_DIRECTORY = "/path/to/your/documentation"
FAISS_INDEX_PATH = "./faiss_index"

# GPU Configuration
USE_FLASH_ATTENTION = True  # Set False on Mac if GPU errors occur
USE_RERANKER = True  # Set False to reduce GPU memory usage
FORCE_MPS = False  # Force MPS on Mac even if CUDA available

# Processing
DEBUG_MODE = False
INCREMENTAL_MODE = True  # Only process new/modified files
BATCH_SIZE_OVERRIDE = None  # None = auto, or specify (1-16)

# Models
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"  # Primary model
FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"

# File Types
SUPPORTED_EXTENSIONS = [".html", ".htm", ".md", ".markdown"]
MAX_FILE_SIZE_MB = 10  # Skip files larger than this

# Ollama Configuration (for image analysis)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llava"  # Model for image analysis
OLLAMA_TIMEOUT = 30  # Seconds

# =============================================================================
# STEP 02 - SEARCH CONFIGURATION (Future)
# =============================================================================

# Search Parameters
DEFAULT_SEARCH_K = 50  # Initial candidates
DEFAULT_RERANK_K = 10  # Final results
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score

# =============================================================================
# STEP 03 - GENERATION CONFIGURATION (Future) 
# =============================================================================

# LLM Configuration
LLM_PROVIDER = "ollama"  # ollama, mlx, transformers
LLM_MODEL = "llama2"  # Model name
LLM_TEMPERATURE = 0.1  # Generation randomness
LLM_MAX_TOKENS = 2048  # Maximum response length

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "localrag.log"

# Performance
MAX_CONCURRENT_FILES = 4  # Parallel file processing
MEMORY_LIMIT_GB = 8  # Stop if memory usage exceeds

# Cache
ENABLE_QUERY_CACHE = True
CACHE_SIZE_MB = 100
CACHE_TTL_HOURS = 24