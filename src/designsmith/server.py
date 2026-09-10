from fastapi import FastAPI, Form, File;
from controllers.user_controller import get_similar_image, GetImageRequest, get_pattern, GetPatternRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile

app = FastAPI()

#configuration in app 
app.add_middleware(
    CORSMiddleware,
    # 1. Allowed origins (Frontend URLs)
    allow_origins=[
       "*" # Production frontend
    ],
    
    # 2. Allowed HTTP methods
    allow_methods=["*"], # Allows all methods (GET, POST, PUT, DELETE, etc.)
    
    # 3. Allowed headers (e.g., Authorization, Content-Type)
    allow_headers=["*"], # Allows all headers
    
    # 4. Allow cookies/auth to be sent cross-origin
    allow_credentials=True, 
)

@app.post("/get_pattern")
#form-data keys must be "query" (text) and "context" (file) — matches the client contract
async def get_pattern_endpoint(query : str = Form(...), context : UploadFile = File(...)):
    print("pattern request received at endpoint")
    return await get_pattern(GetPatternRequest.model_construct(query = query, context = context))

@app.post("/get_similar_image")
#form-data key must be named "context" — matches the client contract 
async def get_similar_image_endpoint(context : UploadFile =  File(...)):
    print("file received at endpoint")
    return await get_similar_image(GetImageRequest.model_construct(context = context))

#health check endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the DesignSmith API!"}
