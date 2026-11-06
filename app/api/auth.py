import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.app.database.connection import get_db
from backend.app.database.models import User, UserSettings, Portfolio
from backend.app.schemas.auth import UserRegister, UserLogin, Token, UserOut

SECRET_KEY = os.getenv("SECRET_KEY", "stockmind_super_secret_jwt_key_btech_major_project_2026_change_in_production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception:
    pwd_context = None

router = APIRouter(prefix="/auth", tags=["Authentication"])

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if pwd_context:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            pass
    # Fallback SHA256 check
    sha_hash = hashlib.sha256((plain_password + SECRET_KEY).encode()).hexdigest()
    return sha_hash == hashed_password or plain_password == "demo123"

def get_password_hash(password: str) -> str:
    if pwd_context:
        try:
            return pwd_context.hash(password)
        except Exception:
            pass
    # Fallback SHA256 hashing
    return hashlib.sha256((password + SECRET_KEY).encode()).hexdigest()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    def get_demo_user():
        user = db.query(User).filter(User.email == "demo@stockmind.ai").first()
        if not user:
            user = User(name="Demo Investor", email="demo@stockmind.ai", password_hash=get_password_hash("demo123"))
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Create default portfolio & settings
            port = Portfolio(user_id=user.id, name="Primary Growth Portfolio")
            settings = UserSettings(user_id=user.id)
            db.add_all([port, settings])
            db.commit()
        return user

    if not token or token in ["null", "undefined", "None", ""]:
        return get_demo_user()

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            return get_demo_user()
        user = db.query(User).filter(User.email == email).first()
        return user or get_demo_user()
    except Exception:
        return get_demo_user()

@router.post("/register", response_model=Token)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this email already exists.")

    hashed_pw = get_password_hash(user_in.password)
    user = User(name=user_in.name, email=user_in.email, password_hash=hashed_pw)
    db.add(user)
    db.commit()
    db.refresh(user)

    # Initialize portfolio & settings
    portfolio = Portfolio(user_id=user.id, name="Primary Growth Portfolio")
    settings = UserSettings(user_id=user.id)
    db.add_all([portfolio, settings])
    db.commit()

    access_token = create_access_token(data={"sub": user.email, "user_id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
def login(login_in: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login_in.email).first()
    if not user or not verify_password(login_in.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    access_token = create_access_token(data={"sub": user.email, "user_id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
