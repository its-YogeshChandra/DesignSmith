from fastapi import FastAPI;

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload")
def upload_file():
    return {"Hello": "World"}   