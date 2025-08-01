import json
import os
from datetime import datetime
from typing import Dict, List, Any
import uuid
from pathlib import Path
import sys

class QueryLogger:
    def __init__(self, log_file: str = "logs/query_logs.json"):
        """
        Initializes the QueryLogger.
        Args:
            log_file (str): Path to the JSON log file.
        """
        self.log_file = Path(log_file)
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        """Ensures the log file and its directory exist with secure permissions."""
        try:
            # Create parent directory with rwxrwxr-x permissions
            self.log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o775)
            if not self.log_file.exists():
                with self.log_file.open('w') as f:
                    json.dump([], f)
                # Set file permissions to rw-rw-r--
                self.log_file.chmod(0o664)
        except Exception as e:
            print(f"Error ensuring log file exists: {e}", file=sys.stderr)
    
    def _read_logs(self) -> List[Dict[str, Any]]:
        """Reads all logs from the log file."""
        try:
            with self.log_file.open('r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_logs(self, logs: List[Dict[str, Any]]):
        """Writes logs to the log file using a safe atomic write pattern."""
        try:
            temp_file = self.log_file.with_suffix('.tmp')
            with temp_file.open('w') as f:
                json.dump(logs, f, indent=2)
            # Set permissions before replacing the original file
            temp_file.chmod(0o664)
            temp_file.replace(self.log_file)
        except Exception as e:
            print(f"Error writing logs: {e}", file=sys.stderr)
    
    def log_query(
        self,
        document_link: str,
        queries: List[str],
        responses: Dict[str, str],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Logs a query, its responses, and metadata to the log file.
        Returns the unique ID of the log entry.
        """
        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "document_link": document_link,
            "queries": queries,
            "responses": responses,
            "metadata": metadata or {}
        }
        
        logs = self._read_logs()
        logs.append(log_entry)
        self._write_logs(logs)
        
        return log_entry["id"]
    
    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent log entries, sorted from newest to oldest.
        """
        logs = self._read_logs()
        # Return a slice of the reversed list
        return logs[-limit:][::-1]

# Create a singleton instance for the application to use
query_logger = QueryLogger()