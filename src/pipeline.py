#goal : index all the images using llamaindex and save it to qdrant 
from llama_cloud import LlamaCloud;
from designsmith.utils.file_utility import download_files_from_s3, list_objects;
import os;
from pathlib import Path;
#get the lama cloud api from the env file 
import requests;

def make_client() -> LlamaCloud:
    cloud = LlamaCloud(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
    return cloud



#function is the main indexing function
#one image at a a time parsing: save from loading too much data on local disk 
#return : image embedding
def indexer(client : LlamaCloud, file_path : str, out_dir: str):
       
    #read file from the file path 
    with open(file_path, "rb") as f:
        #upload file in llama cloud 
        file = client.files.create(file=f, purpose="parse") 

        #parse the file 
        parsing_response = client.parsing.parse(
            file_id=file.id,
            tier= "agentic",
            version="latest",
            output_options= {"images_to_save": ["screenshots", "embedded", "layout" ]},
            expand= ["markdown_full", "images_content_metadata"]
            )

        #download each image via pre assigned url 
        for image in parsing_response.images_content_metadata.images:
            #give the destination path to save the file 
            dest = Path(out_dir) / f"{image.filename}"

            try:
            #fetch image from the llama cloud and send it to the user
            #network io operation : use async  
                image_response = requests.get(image.url,allow_redirects=True, stream=True) 
                with open(dest, "wb") as img_file:
                    for chunk in image_response.iter_content():
                        img_file.write(chunk)
            except ValueError as e:
                print(f"Failed to download image: {e}")

    return {
            "file_destination" : dest,
            "markdown_full" : parsing_response.markdown_full
            }
                



                