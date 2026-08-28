import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import app  # noqa: E402


class DatabaseDeleteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="agenthub-db-delete-tests-")
        self.databases_dir = Path(self.tmp.name) / "databases"
        self.databases_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def create_db(self, name: str) -> Path:
        db_path = self.databases_dir / name
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO probe (name) VALUES (?)", ("temporary",))
            conn.commit()
        finally:
            conn.close()
        return db_path

    def test_delete_database_endpoint_deletes_allowed_workspace_db(self):
        from fastapi.testclient import TestClient

        db_path = self.create_db("delete_me.db")
        with patch.object(config, "DATABASES_DIR", self.databases_dir):
            response = TestClient(app.app).post("/api/databases/delete", json={"db_name": "delete_me.db"})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(db_path.exists())
        self.assertEqual(response.json()["deleted"], "delete_me.db")

    def test_delete_database_endpoint_rejects_traversal(self):
        from fastapi.testclient import TestClient

        outside = Path(self.tmp.name) / "outside.db"
        outside.write_bytes(b"not a workspace database")

        with patch.object(config, "DATABASES_DIR", self.databases_dir):
            response = TestClient(app.app).post("/api/databases/delete", json={"db_name": "../outside.db"})

        self.assertEqual(response.status_code, 400)
        self.assertTrue(outside.exists())


if __name__ == "__main__":
    unittest.main()
