#function to fetch the user input for rag
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from designsmith.utils.rag_utility import embed_image,embed_text, get_embeddings_from_db
from designsmith.utils.file_utility import download_files_from_s3, list_objects;
from fastapi.responses import FileResponse

#class compoenet for getting request 
class GetImageRequest(BaseModel):
    context : UploadFile

#class component for sending response 
class GetImageResponse(BaseModel):
    success: bool
    response : list[FileResponse]

#controller for get image 
# request : the user input 
# response : the response from the rag 
async def get_similar_image(request : GetImageRequest) -> GetImageResponse:
    file_data = request.context
    #read uploaded file bytes  
    raw_bytes = await file_data.read()
    #create embedding from uploaded image using clip model  
    user_query_embedding = embed_image(data=raw_bytes, filename=file_data.filename)

    #retreive similar embeddings from embeddings
    #will doing a nearest neightbor search
    nearest_chunks = get_embeddings_from_db(user_query_embedding)
    
    #fix to the thing
    if nearest_chunks == None:
        return GetImageResponse(success=False, response = "no similar images found")

    #if we find chunks 
    #search the media bucket using image data 
    #iterate over the chunks
    image_response = []
    for chunks in nearest_chunks.points:
        file_name = chunks.payload.get("file_name")
        PROJECT_ROOT = Path(__file__).resolve().parents[1]   # DesignSmith/ even when run from anywhere
        download_destination = str(PROJECT_ROOT / "public" / "embeds")
       
        # retrive file from media bucket
        file_object_path = download_files_from_s3(file_name, download_destination)

        #no mimetype added , fastapi underthe hood adds mimetype : change this in future 
        file_response = FileResponse( path = file_object_path)
        image_response.append(file_response)
        
    return GetImageResponse(success =True, response = image_response)

 