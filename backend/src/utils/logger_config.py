"""
Logging Configuration Module
Provides detailed logging for RAG system operations.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Any
import json


class RAGLogger:
    """Custom logger for RAG system with detailed formatting."""

    def __init__(self, name: str = "RAG_System", level: int = logging.INFO):
        """Initialize the logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Console handler with custom formatting
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)

            # Detailed formatter
            formatter = logging.Formatter(
                '\n{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)

    def log_document_processing(self, filename: str, chunks_count: int, total_words: int):
        """Log document processing details."""
        self.logger.info(f"""
========================================
📄 DOCUMENT PROCESSING
========================================
Filename: {filename}
Total Chunks Created: {chunks_count}
Total Words: {total_words}
Chunk Size: 500 words (configured)
Overlap: 100 words (configured)
========================================
""")

    def log_chunk_details(self, chunk_number: int, chunk_text: str, word_count: int):
        """Log individual chunk details."""
        preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        self.logger.debug(f"""
--- Chunk #{chunk_number} ---
Words: {word_count}
Preview: {preview}
""")

    def log_embedding_generation(self, text_preview: str, embedding_dim: int, task_type: str):
        """Log embedding generation."""
        self.logger.info(f"""
🔢 EMBEDDING GENERATION
Text Preview: {text_preview[:100]}...
Embedding Dimension: {embedding_dim}
Task Type: {task_type}
Model: text-embedding-004
""")

    def log_vector_storage(self, num_vectors: int, collection_name: str):
        """Log vector storage."""
        self.logger.info(f"""
💾 VECTOR STORAGE
Vectors Added: {num_vectors}
Collection: {collection_name}
Database: ChromaDB (Persistent)
""")

    def log_query_start(self, question: str):
        """Log query start."""
        self.logger.info(f"""
========================================
🔍 QUERY PROCESSING STARTED
========================================
Question: {question}
Timestamp: {datetime.now().isoformat()}
========================================
""")

    def log_query_embedding(self, question: str, embedding_sample: List[float]):
        """Log query embedding details."""
        self.logger.info(f"""
🔢 QUERY EMBEDDING GENERATED
Question: {question}
Embedding Dimensions: {len(embedding_sample)}
Sample Values (first 10): {[round(v, 6) for v in embedding_sample[:10]]}
Task Type: retrieval_query
Model: text-embedding-004
""")

    def log_similarity_search(self, top_k: int, results_count: int):
        """Log similarity search."""
        self.logger.info(f"""
🎯 SIMILARITY SEARCH
Requested Top-K: {top_k}
Results Found: {results_count}
Search Method: Cosine Similarity
""")

    def log_retrieved_chunks(self, chunks: List[Dict], distances: List[float] = None):
        """Log retrieved chunks with details."""
        self.logger.info(f"""
📑 RETRIEVED CHUNKS (Context for AI)
========================================
Total Chunks Retrieved: {len(chunks)}
""")

        for i, chunk in enumerate(chunks):
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            distance_info = f"\nSimilarity Distance: {distances[i]:.6f}" if distances and i < len(distances) else ""
            self.logger.info(f"""
--- Chunk {i+1} ---
Preview: {preview}
Length: {len(chunk)} characters{distance_info}
""")

    def log_context_sent_to_ai(self, combined_context: str, word_count: int):
        """Log the full context being sent to AI."""
        self.logger.info(f"""
🤖 CONTEXT SENT TO AI MODEL
========================================
Total Context Length: {len(combined_context)} characters
Estimated Words: {word_count}
Estimated Tokens (approx): {word_count * 1.3:.0f}
Model: gemini-2.0-flash-exp

Context Preview (first 500 chars):
{combined_context[:500]}...
========================================
""")

    def log_prompt_details(self, full_prompt: str, prompt_tokens_estimate: int):
        """Log the full prompt sent to AI."""
        self.logger.info(f"""
📝 FULL PROMPT TO AI
========================================
Prompt Length: {len(full_prompt)} characters
Estimated Prompt Tokens: {prompt_tokens_estimate}

FULL PROMPT:
{full_prompt}
========================================
""")

    def log_ai_response(self, response: str, response_tokens_estimate: int):
        """Log AI response."""
        self.logger.info(f"""
💬 AI RESPONSE RECEIVED
========================================
Response Length: {len(response)} characters
Estimated Response Tokens: {response_tokens_estimate}

RESPONSE:
{response}
========================================
""")

    def log_token_consumption(self, input_tokens: int, output_tokens: int, total_cost_estimate: float):
        """Log token consumption and cost estimate."""
        self.logger.info(f"""
💰 TOKEN CONSUMPTION ESTIMATE
========================================
Input Tokens: {input_tokens}
Output Tokens: {output_tokens}
Total Tokens: {input_tokens + output_tokens}

COST ESTIMATE (Gemini 2.0 Flash):
- Input: ${input_tokens * 0.000000075:.6f} (free tier: < $0.01)
- Output: ${output_tokens * 0.0000003:.6f} (free tier: < $0.01)
- Total: ${total_cost_estimate:.6f}

Note: Actual costs may vary. Check Google AI pricing.
========================================
""")

    def log_sources(self, sources: List[Dict]):
        """Log source documents."""
        self.logger.info(f"""
📚 SOURCE DOCUMENTS
========================================
Total Sources: {len(sources)}
""")
        for i, source in enumerate(sources):
            self.logger.info(f"  {i+1}. {source.get('filename', 'Unknown')} (uploaded: {source.get('upload_date', 'Unknown')})")
        self.logger.info("========================================")

    def log_query_complete(self, total_time_ms: float):
        """Log query completion."""
        self.logger.info(f"""
========================================
✅ QUERY PROCESSING COMPLETED
========================================
Total Processing Time: {total_time_ms:.2f}ms
========================================
""")

    def info(self, message: str):
        """Standard info logging."""
        self.logger.info(message)

    def debug(self, message: str):
        """Standard debug logging."""
        self.logger.debug(message)

    def error(self, message: str):
        """Standard error logging."""
        self.logger.error(message)

    def warning(self, message: str):
        """Standard warning logging."""
        self.logger.warning(message)


# Global logger instance
rag_logger = RAGLogger(level=logging.INFO)
