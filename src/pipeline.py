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

#function is the main indexing function
#uses clip: (image parser) to extract the data 
#one image at a a time parsing: save from loading too much data on local disk 
def parse_image(file_path : str, out_dir: str): 
     #convert image to base64
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

        api_url = os.getenv("CLIP_API_URL")
        api_url = "http://localhost:8000"
        try:
            response = requests.post(
            f"{api_url}/embedding/image",
            json={
                "image": encoded_string 
            }
            ) 
            return response.json().get("detail")[0]; 

        except Exception as e: 
            print("the error is :  ") 
            pprint.pprint(e) 
            
                
#test the parse function 
if __name__ == "__main__":
    import pprint
    result = parse_image("test.jpeg", "out")
    print("the result is : ")
    pprint.pprint(result)

client = QdrantClient(url = "http://localhost:6333");
#create the collection
client.create_collection(
    collection_name="image_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.DOT),
)



#take the index file read the output 
def Store() ->None:
    image_data_storage = list_objects();
     
    for image in image_data_storage: 
       image_data = image.get('Key')
       download_destination = "../public/storage"
       destination_res = download_files_from_s3(image_data, "downloaded")

       output_dir = "../output"   
       parsed_image_data = parse_image(destination_res, output_dir)        
       
       #image embeddings  
       image_data = parse_image

       client.


    
      
                