import json
import os
from datetime import datetime
from typing import Dict, List, Any
import uuid
from pathlib import Path

class QueryLogger:
    def __init__(self, log_file: str = "query_logs.json"):
        """
        Initialize the QueryLogger with a log file path.
        
        Args:
            log_file: Path to the JSON log file. Defaults to 'query_logs.json' in the current directory.
        """
        self.log_file = Path(log_file)
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        """Ensure the log file exists and has a valid JSON array."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
            if not self.log_file.exists():
                with open(self.log_file, 'w') as f:
                    json.dump([], f)
                # Set permissions after file creation
                self.log_file.chmod(0o666)
        except Exception as e:
            import sys
            print(f"Error ensuring log file exists: {e}", file=sys.stderr)
    
    def _read_logs(self) -> List[Dict[str, Any]]:
        """Read and return all logs from the log file."""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_logs(self, logs: List[Dict[str, Any]]):
        """Write logs to the log file."""
        try:
            # Ensure directory exists and has correct permissions
            self.log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
            # Write to a temporary file first, then rename (atomic operation)
            temp_file = self.log_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(logs, f, indent=2)
            # Set permissions and replace the old file
            temp_file.chmod(0o666)
            temp_file.replace(self.log_file)
        except Exception as e:
            import sys
            print(f"Error writing logs: {e}", file=sys.stderr)
            # Try one more time with direct write if atomic replace failed
            try:
                with open(self.log_file, 'w') as f:
                    json.dump(logs, f, indent=2)
                self.log_file.chmod(0o666)
            except Exception as e2:
                print(f"Fallback log write also failed: {e2}", file=sys.stderr)
    
    def log_query(
        self,
        document_link: str,
        queries: List[str],
        responses: Dict[str, str],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Log a query and its responses to the log file.
        
        Args:
            document_link: The URL of the document that was queried
            queries: List of questions that were asked
            responses: Dictionary mapping query numbers to their responses
            metadata: Additional metadata to store with the log entry
            
        Returns:
            str: The unique ID of the log entry
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
        Retrieve the most recent log entries.
        
        Args:
            limit: Maximum number of log entries to return
            
        Returns:
            List of log entries, most recent first
        """
        logs = self._read_logs()
        return logs[-limit:][::-1]  # Return most recent first

# Create a singleton instance for the application
query_logger = QueryLogger("logs/query_logs.json")
