from abc import ABC, abstractmethod
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings
import logging
from typing import BinaryIO
from botocore.config import Config

logger = logging.getLogger(__name__)

class StorageProvider(ABC):
    @abstractmethod
    async def upload_file(self, file_data: bytes, filename: str) -> str:
        pass

    @abstractmethod
    async def get_file_url(self, file_path: str) -> str:
        pass

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        pass

class S3StorageProvider(StorageProvider):
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            config=Config(
                region_name=settings.AWS_REGION,
                signature_version='s3v4'
            )
        )
        self.bucket = settings.AWS_BUCKET_NAME

    async def upload_file(self, file_data: bytes, filename: str) -> str:
        try:
            file_path = f"documents/{filename}"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=file_path,
                Body=file_data,
                ContentType=self._get_content_type(filename)
            )
            return file_path
        except Exception as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            raise

    async def get_file_url(self, file_path: str) -> str:
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': file_path,
                },
                ExpiresIn=3600,
            )
            return url
        except Exception as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            raise

    async def delete_file(self, file_path: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=file_path)
            return True
        except Exception as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False

    def _get_content_type(self, filename: str) -> str:
        ext = filename.split('.')[-1].lower()
        content_types = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'md': 'text/markdown'
        }
        return content_types.get(ext, 'application/octet-stream')