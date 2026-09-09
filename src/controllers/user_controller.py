#function to fetch the user input for rag
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from designsmith.utils.rag_utility import embed_image,embed_text, get_embeddings_from_db
from designsmith.utils.file_utility import download_files_from_s3, list_objects;
from fastapi.responses import FileResponse
from pathlib import Path
import zipfile

#class compoenet for getting request 
class GetImageRequest(BaseModel):
    context : UploadFile


class GetPatternRequest(BaseModel):
    query : str
    context : UploadFile


class StructureResponse(BaseModel): 
   image : UploadFile
   pattern : UploadFile
 

class GetPatternResponse(BaseModel):
    success : bool
    response : StructureResponse
    


#controller for get image 
# request : the user input 
# response : the response from the rag 
async def get_similar_image(request : GetImageRequest) -> FileResponse:
    file_data = request.context
    
    #read uploaded file bytes  
    raw_bytes = await file_data.read()
    
    #create embedding from uploaded image using clip model  
    user_query_embedding = embed_image(data=raw_bytes, filename=file_data.filename)

    #retreive similar embeddings from embeddings
    #will doing a nearest neightbor search
    nearest_chunks = get_embeddings_from_db(user_query_embedding)
    
    #fix to the thing
    if nearest_chunks is None or not nearest_chunks.points:
        raise HTTPException(status_code=404, detail="no similar images found")

    #if we find chunks 
    #search the media bucket using image data 
    #iterate over the chunks
    PROJECT_ROOT = Path(__file__).resolve().parents[1]   # DesignSmith/ even when run from anywhere
    download_destination = PROJECT_ROOT / "public" / "embeds"
    download_destination.mkdir(parents=True, exist_ok=True)

    #matched files are zipped and returned directly : FileResponse can't live inside a pydantic response model
    #client descontruct on the browser side  
    zip_path = download_destination / "similar_images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for chunks in nearest_chunks.points:
            file_name = chunks.payload.get("file_name")
            if file_name is None:
                print(f"Skipping point {chunks.id} — no file_name in payload")
                continue
           
            # retrive file from media bucket
            file_object_path = download_files_from_s3(file_name, str(download_destination))
            zf.write(file_object_path, arcname=file_name)
        
    #no mimetype added , fastapi underthe hood adds mimetype : change this in future 
    return FileResponse(path = str(zip_path), filename = "similar_images.zip", media_type = "application/zip")





#get pattern out 
async def get_pattern(request : GetPatternRequest) -> FileResponse:
    file_data = request.context
    
    #read uploaded file bytes  
    raw_bytes = await file_data.read()
    
    #create embedding from uploaded image using clip model  
    user_query_embedding = embed_image(data=raw_bytes, filename=file_data.filename)

    #retreive similar embeddings from embeddings
    #will doing a nearest neightbor search
    nearest_chunks = get_embeddings_from_db(user_query_embedding)
    
    #fix to the thing
    if nearest_chunks is None or not nearest_chunks.points:
        raise HTTPException(status_code=404, detail="no similar images found")

    #if we find chunks 
    #search the media bucket using image data 
    #iterate over the chunks
    PROJECT_ROOT = Path(__file__).resolve().parents[1]   # DesignSmith/ even when run from anywhere
    download_destination = PROJECT_ROOT / "public" / "embeds"
    download_destination.mkdir(parents=True, exist_ok=True)

    #call the generative llm   
    generated_output = get_llm_response(file_data, nearest_chunks)
    
    #no mimetype added , fastapi underthe hood adds mimetype : change this in future 
    return FileResponse(path = str(zip_path), filename = "similar_images.zip", media_type = "application/zip")
