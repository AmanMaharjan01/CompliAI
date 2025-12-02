"""
Authentication middleware and dependencies
"""

from typing import Dict, List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import os

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """Validate JWT token and return user data"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        role: str = payload.get("role", "employee")
        
        if user_id is None:
            raise credentials_exception
        
        return {
            "user_id": user_id,
            "role": role,
            "email": payload.get("email"),
            "department": payload.get("department")
        }
        
    except JWTError:
        raise credentials_exception


def require_role(allowed_roles: List[str] or str):
    """Dependency to check user role"""
    if isinstance(allowed_roles, str):
        allowed_roles = [allowed_roles]
    
    async def role_checker(current_user: Dict = Depends(get_current_user)) -> Dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {current_user['role']} not authorized. Required: {allowed_roles}"
            )
        return current_user
    
    return role_checker
