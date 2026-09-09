#function to fetch the user input for rag
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from designsmith.utils.rag_utility import embed_image,embed_text, get_embeddings_from_db, get_llm_response, get_mimetype
from designsmith.utils.file_utility import download_files_from_s3, list_objects;
from fastapi.responses import FileResponse
from pathlib import Path
import zipfile
import base64
import uuid

#maximum upload size : 10 mb
#base64 for the llm inflates bytes by ~33% so keep headroom under request limits
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

#how many retrieved images are sent to the vision llm
#llama-3.2-vision accepts a single image per request : send the top match only (points come sorted by similarity)
MAX_LLM_IMAGES = 1

#class compoenet for getting request
class GetImageRequest(BaseModel):
    context : UploadFile


class GetPatternRequest(BaseModel):
    query : str
    context : UploadFile


class GetPatternResponse(BaseModel):
    success : bool
    response : dict


#shared upload validation : empty file , wrong content type , maximum size
def validate_image_upload(raw_bytes: bytes, file_data: UploadFile) -> None:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    if file_data.content_type and not file_data.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"expected an image , got {file_data.content_type}")

    if len(raw_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds maximum size of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB"
                   f" (received {len(raw_bytes) / (1024 * 1024):.1f} MB)",
        )


#controller for get image
# request : the user input
# response : zip of the similar images from the rag
async def get_similar_image(request : GetImageRequest) -> FileResponse:
    file_data = request.context

    #read uploaded file bytes
    raw_bytes = await file_data.read()
    validate_image_upload(raw_bytes, file_data)

    #create embedding from uploaded image using clip model
    user_query_embedding = embed_image(data=raw_bytes, filename=file_data.filename)
    if user_query_embedding is None:
        raise HTTPException(status_code=502, detail="failed to create embedding for the uploaded image")

    #retreive similar embeddings from embeddings
    #will doing a nearest neightbor search
    nearest_chunks = get_embeddings_from_db(user_query_embedding)
    if nearest_chunks is None:
        raise HTTPException(status_code=503, detail="vector search is unavailable")
    if not nearest_chunks.points:
        raise HTTPException(status_code=404, detail="no similar images found")

    #if we find chunks
    #search the media bucket using image data
    #iterate over the chunks
    #parents[2] : user_controller.py sits at src/controllers/ - two levels up is the repo root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]   # DesignSmith/ even when run from anywhere
    download_destination = PROJECT_ROOT / "public" / "embeds"
    download_destination.mkdir(parents=True, exist_ok=True)

    #matched files are zipped and returned directly : FileResponse can't live inside a pydantic response model
    #client decontruct on the browser side
    #unique name per request : concurrent requests must not clobber the same zip
    zip_path = download_destination / f"similar_images_{uuid.uuid4().hex[:8]}.zip"
    files_added = 0
    with zipfile.ZipFile(zip_path, "w") as zf:
        for chunks in nearest_chunks.points:
            file_name = chunks.payload.get("file_name")
            if file_name is None:
                print(f"Skipping point {chunks.id} — no file_name in payload")
                continue

            try:
                # retrive file from media bucket
                file_object_path = download_files_from_s3(file_name, str(download_destination))
                zf.write(file_object_path, arcname=file_name)
                files_added = files_added + 1
            except Exception as e:
                print(f"Skipping {file_name} — download failed: {e}")
                continue

    if files_added == 0:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail="failed to load any of the similar images")

    #no mimetype added , fastapi underthe hood adds mimetype : change this in future
    return FileResponse(path = str(zip_path), filename = "similar_images.zip", media_type = "application/zip")


#get pattern out
# request : the user input image + query
# response : llm generated pattern analysis as json + the matched image names
async def get_pattern(request : GetPatternRequest) -> GetPatternResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")

    file_data = request.context

    #read uploaded file bytes
    raw_bytes = await file_data.read()
    validate_image_upload(raw_bytes, file_data)

    #create embedding from uploaded image using clip model
    user_query_embedding = embed_image(data=raw_bytes, filename=file_data.filename)
    if user_query_embedding is None:
        raise HTTPException(status_code=502, detail="failed to create embedding for the uploaded image")

    #retreive similar embeddings from embeddings
    nearest_chunks = get_embeddings_from_db(user_query_embedding)
    if nearest_chunks is None:
        raise HTTPException(status_code=503, detail="vector search is unavailable")
    if not nearest_chunks.points:
        raise HTTPException(status_code=404, detail="no similar images found")

    #download the matched images and build the llm content as data uris
    #parents[2] : user_controller.py sits at src/controllers/ - two levels up is the repo root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]   # DesignSmith/ even when run from anywhere
    download_destination = PROJECT_ROOT / "public" / "embeds"
    download_destination.mkdir(parents=True, exist_ok=True)

    image_content_list = []
    similar_file_names = []
    for chunks in nearest_chunks.points:
        #cap the number of images sent to the llm
        if len(image_content_list) >= MAX_LLM_IMAGES:
            break

        file_name = chunks.payload.get("file_name")
        if file_name is None:
            print(f"Skipping point {chunks.id} — no file_name in payload")
            continue

        try:
            # retrive file from media bucket and base64 it for the llm
            image_file_path = download_files_from_s3(file_name, str(download_destination))
            with open(image_file_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Skipping {file_name} — download failed: {e}")
            continue

        similar_file_names.append(file_name)
        image_content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{get_mimetype(file_name)};base64,{base64_image}"
            }
        })

    if not image_content_list:
        raise HTTPException(status_code=502, detail="failed to load any of the similar images")

    #call the generative llm with the query + the retrieved images
    generated_output = get_llm_response(query=request.query, image_content_list=image_content_list)
    if generated_output is None:
        raise HTTPException(status_code=502, detail="llm failed to generate a pattern")

    return GetPatternResponse(
        success = True,
        response = {
            "pattern": generated_output.response,
            "similar_images": similar_file_names,
        }
    )
