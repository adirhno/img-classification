import os
from minio.error import S3Error
from PIL import Image
import io

# Function to download and process a JPEG image from MinIO
def download_images_from_minio(client, bucket_name, folder_name, local_dir):
    try:
        # List all objects in the folder
        objects = client.list_objects(bucket_name, prefix=folder_name, recursive=True)
        print(objects)
        for obj in objects:
            if obj.object_name.endswith('.jpg') or obj.object_name.endswith('.jpeg'):
                response = client.get_object(bucket_name, obj.object_name)
            
                img = Image.open(io.BytesIO(response.read()))
                
                local_image_path = os.path.join(local_dir, os.path.basename(obj.object_name))
                
                # Save the image locally
                img.save(local_image_path)
                print(f"Downloaded and saved: {local_image_path}")
    except S3Error as e:
        print(f"Error downloading images from {folder_name}: {e}")


def upload_images_to_minio(client, image_folder, bucket_name):
    for filename in os.listdir(image_folder):
     if filename.endswith(".jpeg") or filename.endswith(".png"):
        file_path = os.path.join(image_folder, filename)
        
        folder_name = "sunglasses" 
        
        object_name = f"{folder_name}/{filename}"
        try:
            # Upload the file to MinIO
            with open(file_path, "rb") as file_data:
                client.put_object(bucket_name, object_name, file_data, os.stat(file_path).st_size)
                print(f"Uploaded {object_name}")
        except S3Error as e:
            print(f"Error uploading {filename}: {e}")


train_dir = '/Users/adir.hino/Desktop/matploit/data/train'
test_dir = '/Users/adir.hino/Desktop/matploit/data/test'

local_train_dir = './data/train/no_sunglasses'
local_test_dir = './data/test/sunglasses'
