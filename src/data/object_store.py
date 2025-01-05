from minio import Minio
import yaml
import os
from typing import BinaryIO
import logging

class ObjectStore:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.client = Minio(
            endpoint=os.environ["MINIO_ADDRESS"],
            access_key=self.config['minio']['access_key'],
            secret_key=self.config['minio']['secret_key'],
            secure=self.config['minio']['secure']
        )
        
        self.bucket_name = self.config['minio']['bucket_name']
        self._ensure_bucket_exists()
        logging.info("Connect to Minio")
        
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)
            
    def upload_file(self, file_path: str, object_name: str = None):
        """Upload a file to MinIO."""
        if object_name is None:
            object_name = os.path.basename(file_path)
            
        self.client.fput_object(
            self.bucket_name,
            object_name,
            file_path
        )
        return object_name
        
    def download_file(self, object_name: str, file_path: str):
        """Download a file from MinIO."""
        self.client.fget_object(
            self.bucket_name,
            object_name,
            file_path
        )
        
    def get_file_url(self, object_name: str, expires=3600):
        """Get a presigned URL for temporary access."""
        return self.client.presigned_get_object(
            self.bucket_name,
            object_name,
            expires=expires
        )