#function to fetch the user input for rag
from fastapi import FastAPI
from pydantic import BaseModel

class GetImageRequest(BaseModel):
    body : str

class GetImageResponse(BaseModel):
    response : str

#controller for get image 
# request : the user input 
# response : the response from the rag 
async def get_image(request : GetImageRequest) -> GetImageResponse:
    user_query = request.body 
    #create user user query embedding using clip model 
    user_query_embedding = get_embedding(user_query)
    
    print(user_query) 
    return GetImageResponse(response = "the response is ")

 