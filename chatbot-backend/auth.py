from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import jwt, JWTError

# --- Configuration ---
# You should load these from environment variables in a real application
SECRET_KEY = "your-super-secret-key"  # Change this to a strong, random key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# --- JWT Token Management ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# --- In-memory User Database (for demonstration purposes) ---
# In a real application, this would be a database (PostgreSQL, SQLite, etc.)
# For now, we store users in a dictionary.
users_db = {} # username: {hashed_password: str, email: str, full_name: Optional[str]}

class UserInDB:
    def __init__(self, username: str, hashed_password: str, email: Optional[str] = None, full_name: Optional[str] = None):
        self.username = username
        self.hashed_password = hashed_password
        self.email = email
        self.full_name = full_name

def get_user(username: str):
    user_data = users_db.get(username)
    if user_data:
        return UserInDB(
            username=username,
            hashed_password=user_data["hashed_password"],
            email=user_data.get("email"),
            full_name=user_data.get("full_name")
        )
    return None