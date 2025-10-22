from twilio.rest import Client

def send_warning(image_url: str, to: str, warning: str):
    client = Client("key", "key")

    message = client.messages.create(
        body=f"Detected equipment: {warning}.",
        from_="whatsapp:number",
        media_url=[image_url],
        to=f"whatsapp:+{to}"
    )

    print("Created message SID:", message.sid)