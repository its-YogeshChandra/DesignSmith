#embedding utility using ai4all/clip api
#provides image and text embedding via CLIP model
import os
import base64
import mimetypes
import subprocess
import requests
from pathlib import Path


CLIP_API_URL = os.getenv("CLIP_API_URL", "http://localhost:8000")


#read a file and return (mime_type, data_uri) for CLIP consumption
def image_to_data_uri(file_path: str) -> tuple[str, str]:
    with open(file_path, "rb") as f:
        raw = f.read()
    mime, _ = mimetypes.guess_type(file_path)
    if mime is None:
        magic = raw[:8]
        stripped = raw.lstrip()
        if magic.startswith(b"\x89PNG"):
            mime = "image/png"
        elif magic.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif magic.startswith(b"GIF8"):
            mime = "image/gif"
        elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            mime = "image/webp"
        elif stripped.startswith(b"<?xml") or stripped.startswith(b"<svg"):
            mime = "image/svg+xml"
        else:
            mime = "image/jpeg"
    data_uri = f"data:{mime};base64," + base64.b64encode(raw).decode()
    return (mime, data_uri)


#get mime type from a filename/key string (no file read needed)
def get_mimetype(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime is None:
        mime = "image/jpeg"
    return mime


#convert SVG to PNG using macOS qlmanage (zero deps)
#returns path to the generated PNG
def svg_to_png(svg_path: str, out_dir: str) -> str:
    subprocess.run(
        ["qlmanage", "-t", "-s", "512", "-o", out_dir, svg_path],
        capture_output=True, check=True
    )
    #qlmanage outputs <filename>.svg.png in the output dir
    png_name = Path(svg_path).name + ".png"
    return str(Path(out_dir) / png_name)


#convert raw bytes + filename to a data URI (for API uploads)
def bytes_to_data_uri(data: bytes, filename: str) -> tuple[str, str]:
    mime = get_mimetype(filename)
    if mime is None:
        #sniff magic bytes as fallback
        if data[:8].startswith(b"\x89PNG"):
            mime = "image/png"
        elif data[:3].startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        else:
            mime = "image/jpeg"
    data_uri = f"data:{mime};base64," + base64.b64encode(data).decode()
    return (mime, data_uri)


#embed an image via CLIP /embedding/image endpoint
#accepts EITHER:
#  file_path (str) — for local files / S3 downloads
#  data (bytes) + filename (str) — for API file uploads 
#handles SVG by converting to PNG first (file_path mode only)
#returns: list of 512 floats (CLIP vector) or None on failure
def embed_image(
    file_path: str = None,
    data: bytes = None,
    filename: str = None,
    temp_dir: str = None
) -> list[float] | None:

    #determine mime and data_uri from either source
    if data is not None and filename is not None:
        #raw bytes from API upload
        mime, data_uri = bytes_to_data_uri(data, filename)
        if mime == "image/svg+xml":
            #SVG upload — save to temp file, convert to PNG, then embed
            import tempfile
            if temp_dir is None:
                temp_dir = tempfile.gettempdir()
            tmp_svg = Path(temp_dir) / filename
            tmp_svg.write_bytes(data)
            try:
                png_path = svg_to_png(str(tmp_svg), temp_dir)
                _, data_uri = image_to_data_uri(png_path)
                Path(png_path).unlink(missing_ok=True)
            except Exception as e:
                print(f"SVG conversion failed for upload {filename}: {e}")
                return None
            finally:
                tmp_svg.unlink(missing_ok=True)
    elif file_path is not None:
        mime = get_mimetype(file_path)
        if temp_dir is None:
            temp_dir = str(Path(file_path).parent)

        if mime == "image/svg+xml":
            #SVG not supported by CLIP — convert to PNG first
            try:
                png_path = svg_to_png(file_path, temp_dir)
                _, data_uri = image_to_data_uri(png_path)
            except Exception as e:
                print(f"SVG conversion failed for {file_path}: {e}")
                return None
        else:
            _, data_uri = image_to_data_uri(file_path)
    else:
        print("embed_image: provide either file_path or (data + filename)")
        return None

    try:
        response = requests.post(
            f"{CLIP_API_URL}/embedding/image",
            json={"images": [data_uri]},
            timeout=120,
        )
        response.raise_for_status()
        vector = response.json()[0]["vector"]

        #cleanup temp PNG if SVG was converted
        if file_path and mime == "image/svg+xml":
            Path(png_path).unlink(missing_ok=True)

        return vector

    except Exception as e:
        print(f"embed_image error: {e}")
        return None


#embed a text string via CLIP /embedding/text endpoint
#returns: list of 512 floats (CLIP vector) or None on failure
def embed_text(text: str) -> list[float] | None:
    try:
        response = requests.post(
            f"{CLIP_API_URL}/embedding/text",
            json={"texts": [text]},
            timeout=120,
        )
        response.raise_for_status()
        vector = response.json()[0]["vector"]
        return vector

    except Exception as e:
        print(f"embed_text error: {e}")
        return None


#embed multiple images in a batch via CLIP /embedding/image endpoint
#returns: list of vectors (each 512 floats) or empty list on failure
def embed_images_batch(file_paths: list[str]) -> list[list[float]]:
    data_uris = []
    for fp in file_paths:
        mime = get_mimetype(fp)
        if mime == "image/svg+xml":
            #skip SVGs in batch — they need individual conversion
            continue
        _, data_uri = image_to_data_uri(fp)
        data_uris.append(data_uri)

    if not data_uris:
        return []

    try:
        response = requests.post(
            f"{CLIP_API_URL}/embedding/image",
            json={"images": data_uris},
            timeout=300,
        )
        response.raise_for_status()
        return [item["vector"] for item in response.json()]

    except Exception as e:
        print(f"embed_images_batch error: {e}")
        return []


#embed multiple text strings in a batch via CLIP /embedding/text endpoint
#returns: list of vectors (each 512 floats) or empty list on failure
def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    try:
        response = requests.post(
            f"{CLIP_API_URL}/embedding/text",
            json={"texts": texts},
            timeout=300,
        )
        response.raise_for_status()
        return [item["vector"] for item in response.json()]

    except Exception as e:
        print(f"embed_texts_batch error: {e}")
        return []


def get_vectordb_client()->QdrantClient:
   client = QdrantClient(url = "http://localhost:6333");
   return client



#function to read and get the embeddings for multiple images
def get_embeddings_from_db(parsed_data: List[float]) -> list[list[float]]:
    client = get_vectordb_client()

    #neartest neighbour search 
    #search based on similarity of vectors
    #future : can add query_filter for particular search 
    response = client.query_points(
        collection_name = "design_smith",
        query = parsed_data,
    )
    #parse response 
    vector = response.points[0].vector  
    return vector 
    
    
        
    