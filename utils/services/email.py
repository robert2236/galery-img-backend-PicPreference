import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

load_dotenv()

def send_email(to_email, subject, body):
    # Configura tu servidor SMTP
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')  
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    sender_email = os.getenv('SENDER_EMAIL', 'tecnorion.it@gmail.com')  
    sender_password = os.getenv('SENDER_PASSWORD', 'default_password') 

    # Crea el mensaje
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Envía el correo
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Inicia la conexión segura
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("Correo enviado exitosamente.")
    except Exception as e:
        print(f"Error al enviar el correo: {e}")