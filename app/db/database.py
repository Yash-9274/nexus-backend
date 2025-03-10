from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError
import time
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
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_timeout=settings.DATABASE_POOL_TIMEOUT,
        pool_pre_ping=True,
        pool_recycle=settings.DATABASE_POOL_RECYCLE,
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
    retries = 3
    retry_delay = 1
    
    for attempt in range(retries):
        try:
            yield db
            break
        except OperationalError as e:
            if attempt == retries - 1:
                logger.error(f"Database connection failed after {retries} attempts")
                raise
            logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
            time.sleep(retry_delay)
        finally:
            db.close()