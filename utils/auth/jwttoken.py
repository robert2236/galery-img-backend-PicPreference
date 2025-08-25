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
def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")  # Extraer user_id del token
        
        if username is None or user_id is None:
            raise credentials_exception
        
        # Construir el objeto User directamente desde el payload
        user = User(
            username=username,
            user_id=user_id,
            email=payload.get("email"),  # Si lo incluiste
            admin=payload.get("admin", False)  # Si lo incluiste
        )
        
        return user
    except JWTError:
        raise credentials_exception
    
