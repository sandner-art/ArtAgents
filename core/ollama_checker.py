# ArtAgent/core/ollama_checker.py
import requests
from typing import Optional
from urllib.parse import urlparse

class OllamaStatusChecker:
    """
    Checks the status of the Ollama service and provides feedback messages.
    Derives the base URL from the provided full API URL.
    """
    def __init__(self, full_api_url: str, timeout: int = 5):
        """
        Initializes the checker.

        Args:
            full_api_url: The full configured Ollama API URL (e.g., "http://host:port/api/generate").
            timeout: The timeout in seconds for the connection attempt.
        """
        self.full_api_url: str = full_api_url
        self.base_url: str = self._derive_base_url(full_api_url) # Derive base URL internally
        self.timeout: int = timeout
        self.available: bool = False
        self.status_message: str = "Check not performed yet."
        self._checked: bool = False

    def _derive_base_url(self, api_url: str) -> str:
        """Internal method to derive the base URL (scheme://netloc)."""
        default_base = "http://localhost:11434" # Default fallback
        try:
            parsed = urlparse(api_url)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}"
            else:
                print(f"Warning: Could not parse scheme/netloc from '{api_url}'. Falling back.")
                return default_base
        except Exception as e:
            print(f"Error parsing Ollama URL '{api_url}': {e}. Falling back.")
            return default_base

    def check(self) -> bool:
        """Performs the connection check to the derived Ollama base URL."""
        self._checked = True
        if not self.base_url:
             self.available = False
             self.status_message = "Error: Base URL could not be determined."
             return False

        try:
            response = requests.get(self.base_url, timeout=self.timeout)
            if response.status_code < 500:
                self.available = True
                self.status_message = f"Ollama responded at base URL {self.base_url} (status: {response.status_code})."
            else:
                self.available = False
                self.status_message = f"Connected, but Ollama server returned status {response.status_code} at {self.base_url}."
        except requests.exceptions.Timeout:
            self.available = False
            self.status_message = f"Connection to Ollama timed out ({self.timeout}s) at {self.base_url}."
        except requests.exceptions.ConnectionError:
            self.available = False
            self.status_message = f"Could not connect to Ollama base URL {self.base_url}. Ensure it's running."
        except requests.exceptions.RequestException as e:
            self.available = False
            self.status_message = f"An unexpected error occurred connecting to Ollama at {self.base_url}. Details: {e}"
        return self.available

    @property
    def is_available(self) -> bool:
        if not self._checked:
             print("Warning: Ollama check has not been performed yet. Call check() first.")
        return self.available

    def get_console_message(self) -> Optional[str]:
        """Generates console message ONLY if unavailable."""
        if self.available:
            return None
        if not self._checked:
            self.status_message = "Check not performed yet."

        return (
            f"\n{'='*60}\n"
            f"🛑 INFO: Ollama Connection Check Failed at Startup! 🛑\n"
            f"{'='*60}\n\n"
            f"ArtAgents requires the Ollama service to be running to process requests.\n\n"
            f"[Details]\n"
            f"  Reason: {self.status_message}\n"
            f"  Derived Base URL Checked: {self.base_url}\n"
            f"  Configured API URL: {self.full_api_url}\n\n"
            f"[Action Suggested]\n"
            f"  If Ollama is not running, please start it using:\n"
            f"    1. The Ollama Desktop application, OR\n"
            f"    2. The command in a *separate terminal*: ollama serve\n\n"
            f"The application UI has loaded, but requests will fail until Ollama is running.\n"
            f"{'='*60}\n"
        )