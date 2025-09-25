import os
import mariadb
import sys
from sqlmodel import Session, create_engine
from config import settings

# Use settings from config
DB_HOST = settings.db_host
DB_USER = settings.db_user
DB_PASSWORD = settings.db_password
DB_NAME = settings.db_name
DB_PORT = settings.db_port

# Use the database_url property from settings
DATABASE_URL = settings.database_url

engine = create_engine(DATABASE_URL, echo=settings.debug)

def get_session():
  with Session(engine) as session:
    yield session

def get_connection():
  try:
    conn = mariadb.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
  except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)
  return conn

# Initialize the database and create the tasks table if it doesn't exist
def init_db():
  conn = get_connection()
  try:
    cursor = conn.cursor()
    
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id INT PRIMARY KEY AUTO_INCREMENT,
        username VARCHAR(255) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL,
        emailaddress VARCHAR(255) NOT NULL UNIQUE,
        token TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    """)

    cursor.execute("""
      CREATE TABLE IF NOT EXISTS user_tokens (
        id INT PRIMARY KEY AUTO_INCREMENT,
        user_id INT NOT NULL,
        token TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        revoked BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
      );
    """)

    cursor.execute("""
      CREATE TABLE IF NOT EXISTS conversations (
        conversation_id INT PRIMARY KEY AUTO_INCREMENT,
        user_id INT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        messages TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
      );
    """)
    
    conn.commit()
    print("Database initialized and tables ensured.")
  except Exception as e:
    print(f"DB init failed: {e}")
  finally:
    conn.close()

init_db()