#the goal is to read the image files from the folder or from the cloudinary api
import requests; 
import boto3;
from dotenv import load_dotenv;
import os;

load_dotenv()

#validate the required var available in environment 
for var in ["CLOUDFLARE_SECRET_KEY", "CLOUDFLARE_ACCESS_KEY", "BUCKET_NAME"]:
    if not os.getenv(var):
        raise ValueError(f"{var} is not set")   


SECRET_KEY = os.getenv("CLOUDFLARE_SECRET_KEY")
ACCESS_KEY = os.getenv("CLOUDFLARE_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

s3 = boto3.client(
    service_name='s3',
    # Provide your R2 endpoint: https://<ACCOUNT_ID>.r2.cloudflarestorage.com
    endpoint_url=f'https://{ACCOUNT_ID}.r2.cloudflarestorage.com',
    
    # Provide your R2 Access Key ID and Secret Access Key
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='auto',  # Required by boto3, not used by R2
)

# Upload a file
#s3.upload_file('myfile.txt', 'my-bucket', 'myfile.txt')
#print('Uploaded myfile.txt')


#download files from the bucket and load them into a folder
# return : path to the file  
def download_files_from_s3(file_name : str):
    s3.download_file(BUCKET_NAME, file_name, file_name)     


# List objects
def list_objects() -> dict[str,any]:
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    return response
