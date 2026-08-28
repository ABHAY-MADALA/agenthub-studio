import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("AGENTHUB_STORAGE_DIR", tempfile.mkdtemp(prefix="agenthub-rag-tests-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.rag_agent import agent, embeddings, ingestion, service, vector_store  # noqa: E402


class RAGResilienceTests(unittest.TestCase):
    def test_upload_uses_local_embeddings_when_openai_embeddings_fail(self):
        with patch.object(type(embeddings._embeddings), "embed_documents", side_effect=RuntimeError("offline")):
            record, is_duplicate = ingestion.ingest_file(
                b"Sick leave policy: employees may request sick leave by emailing their manager.",
                "sick-leave-policy.txt",
            )

        self.assertFalse(is_duplicate)
        self.assertEqual(record.status, "ready")
        self.assertEqual(record.chunk_count, 1)
        self.assertIsNone(record.error)
        self.assertGreaterEqual(vector_store.count(), 1)

    def test_query_returns_grounded_excerpts_when_answer_model_fails(self):
        with patch.object(type(embeddings._embeddings), "embed_documents", side_effect=RuntimeError("offline")):
            ingestion.ingest_file(
                b"Benefits policy: PTO requires manager approval. Sick leave can be requested by email.",
                "benefits-policy.txt",
            )

        with patch.object(type(embeddings._embeddings), "embed_query", side_effect=RuntimeError("offline")), patch.object(
            agent, "_ask_llm", side_effect=RuntimeError("chat offline")
        ):
            result = service.rag_query("What is the sick leave policy?", conversation_id="rag-test-thread")

        self.assertIn("retrieved excerpts", result["answer"])
        self.assertTrue(result["sources"])
        self.assertTrue(result["retrieved_contexts"])


if __name__ == "__main__":
    unittest.main()
