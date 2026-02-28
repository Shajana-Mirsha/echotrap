"""
EchoTrap — Alert System
Logs every detection to alerts.log.
Sends a real SMS via Twilio when a cloned voice is detected.

HOW TO ENABLE SMS:
1. Go to https://twilio.com and create a free account
2. From the Twilio Console get:
   - Account SID  (starts with AC...)
   - Auth Token
   - Your Twilio phone number (e.g. +12025551234)
3. Fill in the three values below and save this file.
   The backend will automatically reload and start sending real SMS alerts.
"""

from datetime import datetime

# ── Twilio credentials ─────────────────────────────────────
# Fill these in with your Twilio Console values.
TWILIO_ACCOUNT_SID  = "YOUR_ACCOUNT_SID"       # e.g. "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN   = "YOUR_AUTH_TOKEN"         # e.g. "your_auth_token_here"
TWILIO_FROM_NUMBER  = "YOUR_TWILIO_NUMBER"      # e.g. "+12025551234"
ALERT_TO_NUMBER     = "YOUR_PHONE_NUMBER"       # e.g. "+919876543210"

SMS_BODY = (
    "EchoTrap Alert: A cloned voice was detected on your family member's phone. "
    "Check on them immediately."
)

# ── Internal helpers ───────────────────────────────────────
def _log(text: str):
    with open("alerts.log", "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")


def _twilio_configured() -> bool:
    return (
        TWILIO_ACCOUNT_SID  != "YOUR_ACCOUNT_SID"  and
        TWILIO_AUTH_TOKEN   != "YOUR_AUTH_TOKEN"   and
        TWILIO_FROM_NUMBER  != "YOUR_TWILIO_NUMBER" and
        ALERT_TO_NUMBER     != "YOUR_PHONE_NUMBER"
    )


# ── Public API ─────────────────────────────────────────────
def send_alert(message: str) -> bool:
    """
    Called by main.py every time a cloned voice is detected.
    Always logs to file. Sends SMS if Twilio credentials are set.
    """
    print(f"[EchoTrap] ALERT: {message}")
    _log(f"ALERT: {message}")

    if not _twilio_configured():
        _log("SMS skipped — Twilio credentials not configured. See alert.py to enable.")
        return True

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=SMS_BODY,
            from_=TWILIO_FROM_NUMBER,
            to=ALERT_TO_NUMBER,
        )
        _log(f"SMS sent successfully. SID: {msg.sid}")
        print(f"[EchoTrap] SMS sent — SID: {msg.sid}")
        return True

    except Exception as e:
        _log(f"SMS failed: {e}")
        print(f"[EchoTrap] SMS error: {e}")
        return False