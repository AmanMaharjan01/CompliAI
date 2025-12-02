"""
Authentication endpoints
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from passlib.context import CryptContext
import os

logger = logging.getLogger(__name__)

router = APIRouter()

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


class UserCreate(BaseModel):
    """User registration model"""
    email: EmailStr
    password: str
    full_name: str
    department: Optional[str] = None


class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str
    expires_in: int


class UserResponse(BaseModel):
    """User data response"""
    user_id: str
    email: str
    full_name: str
    role: str
    department: Optional[str]


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """
    Register new user
    
    - **email**: User email address
    - **password**: User password (min 8 characters)
    - **full_name**: User's full name
    - **department**: Optional department
    """
    # Validate password strength
    if len(user_data.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters"
        )
    
    # Check if user exists (implement database check)
    # For demo, return mock response
    
    hashed_password = get_password_hash(user_data.password)
    
    # Store in database (implement)
    user_id = "user_" + user_data.email.split("@")[0]
    
    return UserResponse(
        user_id=user_id,
        email=user_data.email,
        full_name=user_data.full_name,
        role="employee",
        department=user_data.department
    )


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login and get access token
    
    - **username**: User email
    - **password**: User password
    """
    # Authenticate user (implement database check)
    # For demo purposes, accept any credentials
    
    user_data = {
        "user_id": "demo_user",
        "email": form_data.username,
        "role": "employee"
    }
    
    access_token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(
        data={"sub": user_data["user_id"], "role": user_data["role"]},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_HOURS * 3600
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        user_id=current_user["user_id"],
        email=current_user.get("email", "user@example.com"),
        full_name=current_user.get("full_name", "Demo User"),
        role=current_user["role"],
        department=current_user.get("department")
    )
