from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
from utils import download_images_from_minio, upload_images_to_minio, local_train_dir, local_test_dir
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
load_dotenv()

minio_endpoint = os.getenv('MINIO_ENDPOINT')
minio_port = os.getenv('MINIO_PORT')

client = Minio(
    f"{minio_endpoint}:{minio_port}",
    access_key=os.getenv('MINIO_ACCESS_KEY'),
    secret_key=os.getenv('MINIO_SECRET_KEY'),
)

# Bucket and file details 
bucket_name = "mybucket"
image_folder = "/Users/adir.hino/Desktop/dataset/train/sunglasses"

# Check if the bucket exists, if not create it
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created")

upload_images_to_minio(client, image_folder, bucket_name)

download_images_from_minio(client, bucket_name, 'train/no_sunglasses', local_train_dir)
download_images_from_minio(client, bucket_name, 'test/sunglasses', local_test_dir)