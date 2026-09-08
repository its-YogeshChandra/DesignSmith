#function to fetch the user input for rag
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from designsmith.utils.rag_utility import embed_image,embed_text;

class GetImageRequest(BaseModel):
    context : UploadFile

class GetImageResponse(BaseModel):
    response : str

#controller for get image 
# request : the user input 
# response : the response from the rag 
async def get_image(request : GetImageRequest) -> GetImageResponse:
    file_data = request.context
    #read uploaded file bytes  
    raw_bytes = await file_data.read()
    #create embedding from uploaded image using clip model  
    user_query_embedding = embed_image(data=raw_bytes, filename=file_data.filename)


    #retreive similar embeddings from embeddings  

    print(user_query_embedding) 
    return GetImageResponse(response = "the response is ")

 