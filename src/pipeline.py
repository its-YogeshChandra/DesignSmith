#goal : index all the images using llamaindex and save it to qdrant 
from designsmith.utils.file_utility import download_files_from_s3, list_objects;
import os;
from pathlib import Path;
#get the lama cloud api from the env file 
import requests;
from liteparse import LiteParse;
from llama_index.core import VectorStoreIndex;
import base64;
from qdrant_client import QdrantClient;
from qdrant_client.models import PointStruct, VectorParams, Distance;
import mimetypes;
import uuid;

#function is the main indexing function
#uses clip: (image parser) to extract the data 
#one image at a a time parsing: save from loading too much data on local disk 

def image_data_uri(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read()
    mime, _ = mimetypes.guess_type(file_path)
    if mime is None:  # unknown extension — sniff the magic bytes
        magic = raw[:8]
        if magic.startswith(b"\x89PNG"):
            mime = "image/png"
        elif magic.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif magic.startswith(b"GIF8"):
            mime = "image/gif"
        elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()

#input : object key 
#output : string of mime type
def image_mimetype(object_key : str)-> str:
    mime, _ = mimetypes.guess_type(object_key)
    if mime is None:
        mime = "image/jpeg"  # default fallback
    return mime




def parse_image(file_path : str, out_dir: str): 
    #convert image to base64
    image_string = image_data_uri(file_path)
    api_url = os.getenv("CLIP_API_URL")
    api_url = "http://localhost:8000"
    try:
        response = requests.post(
            f"{api_url}/embedding/image",
            json={"images": [image_string]},
            timeout=120,
            )
        response.raise_for_status()
        vector = response.json()[0]["vector"]  # 512 floats
        return vector
    
    except Exception as e: 
        print("the error is :  ") 
        pprint.pprint(e) 
            
                
#test the parse function 
# if __name__ == "__main__":
#     import pprint
#     result = parse_image("test.jpeg", "out")
#     print("the result is : ")
#     pprint.pprint(result)


client = QdrantClient(url = "http://localhost:6333");
#create the collection
client.create_collection(
    collection_name="image_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.DOT),
)



#take the index file read the output 
def Store() ->None:
    image_data_storage = list_objects();
    print("the image data storage is : ", image_data_storage)     
    points = []
    for image in image_data_storage: 
       image_data = image.get('Key')
       image_mimetype_res = image_mimetype(image_data) 
       
       download_destination = "../public/storage"
       destination_res = download_files_from_s3(image_data, "downloaded")

       output_dir = "../output"   
       parsed_image_data = parse_image(destination_res, output_dir)        
       
       #image embeddings 
       #iterate over the parse images and create points vector  
       points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=parsed_image_data,
                payload={
                    "file_name" : image_data,
                    "mimetype" : image_mimetype_res,
                }
                
            )
          )

    try:      
        qdrant_info = client.upsert(
            collection_name= "image_collection",
            wait = True,
            points = points
        )
        print("the qdrant info is : ", qdrant_info)

    except Exception as e: 
        print("the error is :  ") 
        pprint.pprint(e) 

    
      
                