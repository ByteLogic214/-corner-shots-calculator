"""Telegram alerting integration."""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Lightweight wrapper around the Telegram Bot API."""

    DEFAULT_TIMEOUT: int = 10
    API_BASE: str = "https://api.telegram.org/bot{token}"

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialise the notifier.

        Args:
            token: Bot token. Falls back to TELEGRAM_BOT_TOKEN env var.
            chat_id: Target chat ID. Falls back to TELEGRAM_CHAT_ID env var.
            timeout: HTTP timeout in seconds.
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Return True if both token and chat_id are present."""
        return bool(self.token and self.chat_id)

    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a text message to the configured chat.

        Args:
            message: Text payload (Markdown supported by default).
            parse_mode: Telegram parse mode (Markdown, HTML, etc.).

        Returns:
            True if the message was accepted by Telegram (HTTP 200), else False.
        """
        if not self.is_configured():
            logger.debug("Telegram notifier not configured; skipping alert.")
            return False

        url = f"{self.API_BASE.format(token=self.token)}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.warning("Failed to send Telegram alert: %s", exc)
            return False


def send_telegram_alert(message: str) -> bool:
    """Convenience function using environment-based configuration.

    Args:
        message: Message text to send.

    Returns:
        True on success, False otherwise.
    """
    notifier = TelegramNotifier()
    return notifier.send_message(message)
