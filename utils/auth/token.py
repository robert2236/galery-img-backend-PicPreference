import random
import string
import datetime

# Lista de tokens válidos con sus tiempos de expiración
valid_tokens = {}

def generate_short_token(length=8):
    # Genera un token aleatorio de 8 caracteres alfanuméricos
    characters = string.ascii_letters + string.digits  # Letras y dígitos
    token = ''.join(random.choices(characters, k=length))
    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
    valid_tokens[token] = expiration_time
    return token

def verify_short_token(token: str):
    if token in valid_tokens:
        if valid_tokens[token] > datetime.datetime.utcnow():
            return True
        else:
            del valid_tokens[token]
    return False