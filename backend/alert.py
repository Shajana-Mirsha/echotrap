def send_alert(message):
    try:
        # For now printing the alert
        # We will connect Twilio SMS at the end
        print(f"ECHOTRAP ALERT TRIGGERED: {message}")
        
        # Store alert in a local log file
        with open("alerts.log", "a") as f:
            f.write(f"ALERT: {message}\n")
        
        return True
    except Exception as e:
        print(f"Alert failed: {e}")
        return False