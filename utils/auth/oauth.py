from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from utils.auth.jwttoken import verify_token
from jose import JWTError, jwt
from dotenv import load_dotenv
import os

load_dotenv()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'default_secret_key')
ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    return verify_token(token, credentials_exception)

async def extract_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(
            token,
            "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7",  # Tu SECRET_KEY
            algorithms=["HS256"],
            options={"verify_signature": True}  # Mantén la verificación de seguridad
        )
        
        # Usamos 'sub' como user_id (estándar JWT)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=400, detail="Campo 'sub' no encontrado en el token")
            
        return {"user_id": user_id}  # Devuelve el username como user_id
        
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Token inválido: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )