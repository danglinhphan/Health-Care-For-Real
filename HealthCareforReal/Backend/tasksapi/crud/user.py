from sqlmodel import Session, select, SQLModel, Field, delete
from db.db import engine
import bcrypt
from typing import Optional
from datetime import datetime, timedelta

class User(SQLModel, table=True):
    __tablename__ = "users"
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    password: str
    emailaddress: str
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

class UserLogin(SQLModel):
    username: str
    password: str

class UserCreate(SQLModel):
    username: str
    password: str
    emailaddress: str

class UserUpdate(SQLModel):
    password: str
    emailaddress: str

class UserToken(SQLModel, table=True):
    __tablename__ = "user_tokens"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int
    token: str
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    revoked: Optional[bool] = False

def create_user(user: UserCreate) -> dict:
    print(f"Creating user: {user.username}")
    cleanup_expired_tokens()
    password_hash = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    with Session(engine) as session:
        db_user = User(
            username=user.username, 
            password=password_hash, 
            emailaddress=user.emailaddress
        )
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        print(f"User created successfully with ID: {db_user.id}")
        return {"user_id": db_user.id, "username": db_user.username, "emailaddress": db_user.emailaddress}

def get_user_by_username(username: str) -> dict | None:
    print(f"Looking for user: {username}")
    cleanup_expired_tokens()
    with Session(engine) as session:
        statement = select(User).where(User.username == username)
        result = session.exec(statement).first()
        if result:
            print(f"Found user in database: {username}")
            return {
                "user_id": result.id,
                "username": result.username,
                "password": result.password,
                "emailaddress": result.emailaddress
            }
        print(f"No user found in database: {username}")
        return None

def verify_user_login(username: str, password: str) -> dict | None:
    print(f"Attempting login for username: {username}")
    cleanup_expired_tokens()
    user = get_user_by_username(username)
    if not user:
        print(f"User not found: {username}")
        return None
    print(f"User found: {user['username']}")
    if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        print(f"Password verified for user: {username}")
        return {
            "user_id": user['user_id'],
            "username": user['username'],
            "emailaddress": user['emailaddress']
        }
    print(f"Password verification failed for user: {username}")
    return None

def cleanup_expired_tokens() -> bool:
    print("Cleaning up expired tokens")
    with Session(engine) as session:
        now = datetime.utcnow()
        statement = select(UserToken).where(UserToken.expires_at != None, UserToken.expires_at < now)
        expired_tokens = session.exec(statement).all()
        for token in expired_tokens:
            token.revoked = True
            session.add(token)
        session.commit()
        print(f"Cleaned up {len(expired_tokens)} expired tokens")
        return True

def get_user_token(user_id: int) -> str | None:
    print(f"Getting token for user_id: {user_id}")
    cleanup_expired_tokens()
    with Session(engine) as session:
        now = datetime.utcnow()
        statement = select(UserToken).where(
            UserToken.user_id == user_id,
            (UserToken.expires_at == None) | (UserToken.expires_at > now),
            UserToken.revoked == False
        ).order_by(UserToken.created_at.desc())
        result = session.exec(statement).first()
        if result:
            print(f"Found active token for user_id: {user_id}")
            return result.token
        print(f"No active token found for user_id: {user_id}")
        return None

def save_user_token(user_id: int, token: str) -> bool:
    print(f"Saving token for user_id: {user_id}")
    cleanup_expired_tokens()
    with Session(engine) as session:
        session.exec(delete(UserToken).where(UserToken.user_id == user_id, UserToken.revoked == False))
        expires_at = datetime.utcnow() + timedelta(days=1)
        user_token = UserToken(user_id=user_id, token=token, expires_at=expires_at)
        session.add(user_token)
        session.commit()
        print(f"Token saved successfully in user_tokens table: True")
        return True

def clear_user_token(user_id: int) -> bool:
    print(f"Deleting token for user_id: {user_id}")
    cleanup_expired_tokens()
    with Session(engine) as session:
        statement = select(UserToken).where(UserToken.user_id == user_id, UserToken.revoked == False)
        tokens = session.exec(statement).all()
        for token in tokens:
            session.delete(token)
        session.commit()
        print(f"Deleted token(s) for user_id: {user_id}")
        return True

def get_user_by_id(user_id: int) -> dict | None:
    """Get user by ID for JWT token verification"""
    print(f"Looking for user by ID: {user_id}")
    cleanup_expired_tokens()
    with Session(engine) as session:
        statement = select(User).where(User.id == user_id)
        result = session.exec(statement).first()
        if result:
            print(f"Found user by ID: {user_id}")
            return {
                "user_id": result.id,
                "username": result.username,
                "emailaddress": result.emailaddress
            }
        print(f"No user found by ID: {user_id}")
        return None

def update_user(user_id: int, user: UserUpdate) -> dict:
    print(f"Updating user: {user_id}")
    cleanup_expired_tokens()
    with Session(engine) as session:
        statement = select(User).where(User.id == user_id)
        db_user = session.exec(statement).first()
        if db_user:
            password_hash = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            db_user.password = password_hash
            db_user.emailaddress = user.emailaddress
            session.add(db_user)
            session.commit()
            print(f"User updated successfully: True")
            return {"updated": True}
        print(f"User not found for update: {user_id}")
        return {"updated": False}
