from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from dotenv import load_dotenv
import os

from database.databases import user

load_dotenv()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'default_secret_key')
ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')


def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
    except JWTError:
        raise credentials_exception

    if not username:
        raise credentials_exception

    db_user = await user.find_one({"username": username})
    if db_user is None:
        raise credentials_exception

    db_user["_id"] = str(db_user["_id"])
    return db_user


async def extract_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)

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
