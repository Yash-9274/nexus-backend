from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from sqlalchemy.pool import QueuePool
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    # Get the database URL
    db_url = settings.DATABASE_URL
    logger.info("Initializing database connection...")

    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={
            "sslmode": "require",
            "connect_timeout": 60
        }
    )
    
    # Test the connection using SQLAlchemy text()
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        logger.info("Database connection successful")

except Exception as e:
    logger.error(f"Database connection error: {str(e)}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()