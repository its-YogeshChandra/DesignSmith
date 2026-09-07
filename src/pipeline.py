#goal : index all the images using llamaindex and save it to qdrant 
from designsmith.utils.file_utility import download_files_from_s3, list_objects;
import os;
from pathlib import Path;
#get the lama cloud api from the env file 
import requests;
from liteparse import LiteParse;



#function is the main indexing function
#one image at a a time parsing: save from loading too much data on local disk 
#return : parsed response from liteparse 
def parse(file_path : str, out_dir: str): 
    #parse locally using liteparse (no Docker needed)
    parser = LiteParse(
        output_format="json",
        extract_images=True,
        extract_links=True
    )
    response = parser.parse(file_path)
     
    return {
            "file_path" : file_path,
            "response" : response
            }
                
#test the parse function 
if __name__ == "__main__":
    import pprint
    result = parse("(21) X.jpeg", "out")
    pprint.pprint(result)


#take the index file read the output 
def embedder() ->None:
    image_data_storage = list_objects();
     
    for image in image_data_storage: 
       image_data = image.get('Key')
       download_destination = "../public/storage"
       destination_res = download_files_from_s3(image_data, "downloaded")

       output_dir = "../output"   
       parser_client = parser_client() 
       
       #parse the image data 
       parse_res = parse(destination_res, output_dir, parser_client)

       #chunk the parsed response  

    
      
                