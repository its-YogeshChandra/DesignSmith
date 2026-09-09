#function to fetch the user input for rag
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from designsmith.utils.rag_utility import embed_image,embed_text, get_embeddings_from_db;

class GetImageRequest(BaseModel):
    context : UploadFile

class GetImageResponse(BaseModel):
    success: bool
    response : File

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
    #iterate over the cunks 
    #for val in nearest_chunks:
         
    print(user_query_embedding) 
    return GetImageResponse(response = "the response is ")

 