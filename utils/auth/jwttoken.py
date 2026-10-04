from datetime import datetime, timedelta
from jose import JWTError, jwt
from models.users import TokenData,User
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'default_secret_key')
ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 120))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
def verify_token(token: str, credentials_exception) -> str:
    """Valida la firma/expiración del JWT y devuelve el username (claim 'sub')."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if not username:
            raise credentials_exception

        return username
    except JWTError:
        raise credentials_exception
    
