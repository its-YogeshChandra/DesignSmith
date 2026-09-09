from fastapi import FastAPI;
from .controllers.user_controller import get_similar_image

app = FastAPI()

#configuration in app 


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/get_similar_image")
async def get_similar_image():
    return await get_similar_image()

 