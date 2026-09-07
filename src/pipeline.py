#goal : index all the images using llamaindex and save it to qdrant 
from designsmith.utils.file_utility import download_files_from_s3, list_objects;
import os;
from pathlib import Path;
#get the lama cloud api from the env file 
import requests;
from liteparse import LiteParse;

def make_parser_client() -> LiteParse:
    parser_client = LiteParse(
    ocr_enabled= True,
    ocr_server_url="http://localhost:8001",
    output_format="json",
    extract_images=True,
    extract_links=True
    )
    return parser_client

#function is the main indexing function
#one image at a a time parsing: save from loading too much data on local disk 
#return : image embedding
def parse_and_chunk(file_path : str, out_dir: str, parser_client: LiteParse): 
    #read file from the file path 
    with open(file_path, "rb") as f:
        #upload file in llama cloud 
        data_bytes = f.read()
        response = parser_client.parse(data=data_bytes)
        
        #download each image via pre assigned url   
    return {
            "file_destination" : dest,
            "response" : parsing_response
            }
                
#test the indexer function 
print(indexer("test.pdf", "out", make_parser_client()))

                