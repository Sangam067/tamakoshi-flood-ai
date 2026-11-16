"""SMTP email notifications for high-risk Tamakoshi forecasts."""

import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional

import streamlit as st


class EmailConfigurationError(ValueError):
    """Raised when SMTP settings are missing or invalid."""


@dataclass(frozen=True)
class EmailSettings:
    smtp_host: str
    smtp_port: int
    sender_email: str
    app_password: str
    recipient_email: str
    enabled: bool


def get_email_settings() -> Optional[EmailSettings]:
    try:
        config = st.secrets["email"]
    except (FileNotFoundError, KeyError):
        return None

    required = ("smtp_host", "smtp_port", "sender_email", "app_password", "recipient_email")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise EmailConfigurationError(
            "Email settings are incomplete. Missing: " + ", ".join(missing)
        )

    try:
        port = int(config["smtp_port"])
    except (TypeError, ValueError) as exc:
        raise EmailConfigurationError("email.smtp_port must be a valid port number.") from exc
    if not 1 <= port <= 65535:
        raise EmailConfigurationError("email.smtp_port must be between 1 and 65535.")

    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise EmailConfigurationError("email.enabled must be true or false.")

    return EmailSettings(
        smtp_host=str(config["smtp_host"]),
        smtp_port=port,
        sender_email=str(config["sender_email"]),
        app_password=str(config["app_password"]),
        recipient_email=str(config["recipient_email"]),
        enabled=enabled,
    )


def _send_email(settings: EmailSettings, subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = settings.sender_email
    message["To"] = settings.recipient_email
    message["Subject"] = subject
    message.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
        smtp.ehlo()
        smtp.starttls(context=context)
        smtp.ehlo()
        smtp.login(settings.sender_email, settings.app_password)
        smtp.send_message(message)


def _alert_body(forecast: dict, synced_at: str, source: str) -> str:
    return (
        "TAMAKOSHI HIGH-RISK FORECAST\n\n"
        "The model has entered RED ALERT. Check current river conditions and follow "
        "instructions from local disaster-management authorities.\n\n"
        f"Predicted next-day flow: {forecast['pred_flow']:.2f} m³/s\n"
        f"Danger threshold: {forecast['danger_threshold']:.2f} m³/s\n"
        f"Flood probability: {forecast['flood_prob']:.1f}%\n"
        f"Telemetry source: {source}\n"
        f"Data synced: {synced_at}\n\n"
        "This automated model output is decision support, not an official flood warning."
    )


def demo_email_content(forecast: dict) -> tuple[str, str]:
    subject = "DEMO ONLY - Tamakoshi Flood Risk Email Test"
    body = (
        "This is a demonstration email initiated manually from the Tamakoshi app. "
        "It is not a real warning and was not triggered by a high-risk event.\n\n"
        f"Current model classification: {forecast['alert_level']}\n"
        f"Predicted next-day flow: {forecast['pred_flow']:.2f} m³/s\n"
        f"Danger threshold: {forecast['danger_threshold']:.2f} m³/s\n"
        f"Flood probability: {forecast['flood_prob']:.1f}%\n\n"
        "This automated model output is decision support, not an official flood warning."
    )
    return subject, body


def send_demo_email(forecast: dict) -> str:
    settings = get_email_settings()
    if settings is None:
        raise EmailConfigurationError(
            "No email secrets are configured. Add the values from "
            ".streamlit/secrets.toml.example; no email was sent."
        )
    subject, body = demo_email_content(forecast)
    _send_email(settings, subject, body)
    return settings.recipient_email


def send_live_red_alert_once(forecast: dict, source_mode: str, synced_at: str) -> Optional[str]:
    is_live_red = (
        source_mode == "LIVE_API"
        and forecast.get("alert_level") == "RED_ALERT"
    )
    if not is_live_red:
        st.session_state["tamakoshi_red_email_attempted"] = False
        return None

    if st.session_state.get("tamakoshi_red_email_attempted", False):
        return None

    try:
        settings = get_email_settings()
    except EmailConfigurationError as exc:
        return f"High-risk email is not configured correctly: {exc}"
    if settings is None or not settings.enabled:
        return None

    # Record before sending so reruns cannot repeatedly email during the same red event.
    st.session_state["tamakoshi_red_email_attempted"] = True
    try:
        _send_email(
            settings,
            "HIGH RISK - Tamakoshi River Flood Forecast",
            _alert_body(forecast, synced_at, "Live Open-Meteo telemetry"),
        )
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        return f"High-risk email could not be sent: {exc}"

    return f"High-risk email sent to {settings.recipient_email}."
