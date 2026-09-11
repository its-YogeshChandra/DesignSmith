#embedding utility using ai4all/clip api
#provides image and text embedding via CLIP model
import os
import base64
import json
import mimetypes
import subprocess
import requests
from pathlib import Path
from qdrant_client.models import QueryResponse
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from google import genai
client = genai.Client()
load_dotenv();

#connect to clip url 
CLIP_API_URL = os.getenv("CLIP_API_URL", "http://localhost:8000")

#nvidia nim api key
#validated when the llm is called , not at import : a missing key must not stop the embedding endpoints from running
LLM_API_KEY = os.getenv("LLM_API_KEY")

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
#future work : add similarity score for the images 
#returns QueryResponse on success , None if the db is unreachable or the query fails
def get_embeddings_from_db(parsed_data: list[float]) -> QueryResponse | None:
    if parsed_data is None:
        print("get_embeddings_from_db: no embedding provided")
        return None

    client = get_vectordb_client()

    #nearest neighbour search
    #search based on similarity of vectors
    #future : can add query_filter for particular search
    try:
        response = client.query_points(
            collection_name = "image_collection",
            query = parsed_data,
        )
    except Exception as e:
        print(f"get_embeddings_from_db error: {e}")
        return None

    #parse response
    return response


#system prompt : how the design assistant should behave and shape its output
#hardened after live testing : llama-3.2-11b-vision ignores soft "respond with json"
#wording and returns markdown prose - the instruction must lead , show the exact
#shape , and bound the output ("starts with { ends with }")
LLM_SYSTEM_PROMPT = (
    "You are DesignSmith's design assistant. "
    "You are given reference images retrieved from a design library because they are visually "
    "similar to the user's uploaded image, together with the user's query. "
    "Analyse the reference images and answer the query. "
    "Output ONLY a JSON object. No prose, no markdown, no code fences. "
    "Exact shape: "
    '{"summary": "<one paragraph overview of the visual pattern>", '
    '"style_tags": ["<short style descriptors>"], '
    '"dominant_colors": ["<hex color codes>"], '
    '"recommendations": ["<actionable design suggestions>"]} '
    "Your entire response must start with { and end with }."
)


#vision llm served by nvidia nim
#neva-22b and vila are listed by /v1/models but their backend functions are not available for this account (404)
#llama-3.2-vision works but accepts only ONE image per request - callers must send at most one
LLM_MODEL = "meta/llama-3.2-11b-vision-instruct"


#holds the text output of the llm
class LLMResponse:
    def __init__(self, response: str):
        self.response = response


#llms are probabilistic : even with the hardened prompt the model sometimes wraps
#the json in prose or markdown fences - pull the {...} block out so callers always
#get a parseable json string , or None if nothing json-shaped came back
def _extract_json_block(text: str) -> str | None:
    cleaned = text.strip()

    #strip a markdown code fence if present (```json ... ```)
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = cleaned[start:end + 1]
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


#call the nvidia nim vision llm (neva-22b) with the retrieved images + the user query
#image_content_list : entries of the openai content form {"type": "image_url", "image_url": {"url": "<data uri>"}}
#returns LLMResponse on success , None on failure
def get_llm_response(query: str, image_content_list: list[dict]) -> LLMResponse | None:
    if not LLM_API_KEY:
        print("get_llm_response: LLM_API_KEY is not set")
        return None
    if not query or not query.strip():
        print("get_llm_response: query is empty")
        return None
    if not image_content_list:
        print("get_llm_response: no image content provided")
        return None

    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Accept": "application/json",
    }

    #user content : the query text first (with the json constraint repeated -
    #small vision models obey the last instruction they read) , then the images
    user_content = [{"type": "text", "text": f"{query}\n\nReply with ONLY the JSON object."}]
    for image_entry in image_content_list:
        user_content.append(image_entry)

    payload = {
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "model": LLM_MODEL,
        "max_tokens": 512,
        "seed": 0,
        "stream": False,
        #low temperature : creative sampling drifts into markdown prose , breaking the json contract
        "temperature": 0.2,
        "reasoning_effort": "low",
    }

    try:
        response = requests.post(invoke_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_body = response.json()

        #non streaming : the generated text sits in choices[0].message.content
        generated_text = response_body["choices"][0]["message"]["content"]
        if not generated_text or not generated_text.strip():
            print("get_llm_response: llm returned empty content")
            return None

        #prefer the model's own json ; if it drifted into prose/fences , extract the {...} block
        generated_text = generated_text.strip()
        try:
            json.loads(generated_text)
        except json.JSONDecodeError:
            extracted = _extract_json_block(generated_text)
            if extracted is None:
                print(f"get_llm_response: no parseable json in llm output: {generated_text[:150]}")
                return None
            generated_text = extracted

        return LLMResponse(response=generated_text)

    except requests.HTTPError as e:
        #print the body too : nim errors (bad model , multi-image , auth) are only readable there
        error_body = e.response.text[:300] if e.response is not None else ""
        print(f"get_llm_response error: {e} | body: {error_body}")
        return None
    except Exception as e:
        print(f"get_llm_response error: {e}")
        return None


def ts_llm(query: str, image_content_list: list[dict]) -> LLMResponse | None:
    if not LLM_API_KEY:
        print("ts_llm: LLM_API_KEY is not set")
        return None
        
    if not query or not query.strip():
        print("ts_llm: query is empty")
        return None
       
    if not image_content_list:
        print("ts_llm: no image content provided")
        return None
       
        client = genai.Client()
        
        #call the llm api for the image
        gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview"
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
 
        if not GEMINI_API_KEY:
            print("ts_llm: GEMINI_API_KEY is not set")
            return None

        #build the request body for the gemini api
        interactions = client.interactions.create(
            model = "gemini-3.8-flash",
            input = [ 
                {"type": "text", "text": f"Compare this local image and this remote audio file."},
                {"type": "image", "data": image_b64, "mime_type": f"{image_mime_type}"},
                ]
        )
        request_body = {
            "contents": [
                {
                    "parts": image_content_list + [{"text": query}]
                }
            ]
        }

             
