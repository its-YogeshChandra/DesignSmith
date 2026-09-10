"""
Tests for the api layer : server.py routes + user_controller.py logic

Covers three groups :
  1. POSITIVE  - the flows we want to pass (happy paths)
  2. NEGATIVE  - requests that must fail in a specific way (exact status + detail)
  3. SECURITY  - documented comments for tests that SHOULD be added once the
                 protections exist (not written as code - the api lacks them today)

External services (CLIP :8000 , Qdrant :6333 , NVIDIA NIM , R2) are never hit -
embed_image / get_embeddings_from_db / download_files_from_s3 / get_llm_response
are replaced per-test via monkeypatch on the controller module namespace.

Run with:  uv run pytest tests/test_api.py -v
"""
import sys, os, io, json, struct, zlib, zipfile
from pathlib import Path
from types import SimpleNamespace

# Add src/ to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from fastapi.testclient import TestClient

from designsmith.server import app
import controllers.user_controller as controller
from designsmith.utils.rag_utility import LLMResponse

client = TestClient(app, raise_server_exceptions=False)


# ── helpers ────────────────────────────────────────────────────────

#minimum fake embedding : any list of floats satisfies the stubbed chain
FAKE_VECTOR = [0.1] * 512


#build a tiny REAL png (8x8 single color) with the stdlib
#why : the upload path treats bytes as an image from the very first validation ,
#so tests should send structurally valid image bytes , not b"xxx"
def _make_png_bytes() -> bytes:
    width = height = 8
    raw_rows = b""
    for _ in range(height):
        raw_rows = raw_rows + b"\x00" + b"\x11\x22\x33" * width

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw_rows)) + _chunk(b"IEND", b""))


PNG_BYTES = _make_png_bytes()


#build fake qdrant points matching what the controller reads
#(payload["file_name"] , .id used in skip logs , .score from similarity search)
def _make_points(*file_names):
    points = []
    for name in file_names:
        payload = {} if name is None else {"file_name": name}
        points.append(SimpleNamespace(id=f"id-{name}", payload=payload, score=0.9))
    return SimpleNamespace(points=points)

#money patching : 
#checking if controller working fine or not / not the actual functions that are getting called in them 

#standard stub set : embedding works , db returns points , s3 download writes a
#real temp file (controller opens that path for b64/zip) , llm returns valid json
def _install_happy_stubs(monkeypatch, tmp_path, points, llm_response=None):
    def fake_embed(data, filename):
        return FAKE_VECTOR

    def fake_db(vector):
        return points

    def fake_download(file_name, dest_folder):
        dest = tmp_path / file_name
        dest.write_bytes(PNG_BYTES)
        return str(dest)

    def fake_llm(query, image_content_list):
        return llm_response or LLMResponse(response=json.dumps({
            "summary": "a clean minimalist layout",
            "style_tags": ["minimal"],
            "dominant_colors": ["#111111"],
            "recommendations": ["increase contrast"],
        }))

    
    monkeypatch.setattr(controller, "embed_image", fake_embed)
    monkeypatch.setattr(controller, "get_embeddings_from_db", fake_db)
    monkeypatch.setattr(controller, "download_files_from_s3", fake_download)
    monkeypatch.setattr(controller, "get_llm_response", fake_llm)


# ════════════════════════════════════════════════════════════════════
# 1. POSITIVE TESTS — these must pass
# ════════════════════════════════════════════════════════════════════

#what : liveness of the app object - GET / returns the hello payload
#why   : cheapest proof the app imports , routes register and the harness works
def test_root_returns_hello():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to DesignSmith API!"}


#what : /get_similar_image full happy path - real form upload , stubbed rag chain ,
#       response is a zip that actually contains every matched file
#why   : proves route -> controller -> zip wiring end to end ; if the field name ,
#        the zip writing or the FileResponse contract breaks , this fails
def test_get_similar_image_returns_zip_with_matches(monkeypatch, tmp_path):
    _install_happy_stubs(monkeypatch, tmp_path, _make_points("a.png", "b.png"))

    response = client.post(
        "/get_similar_image",
        files={"context": ("upload.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert sorted(zf.namelist()) == ["a.png", "b.png"]


#what : /get_pattern full happy path - query + image in , llm json out
#why   : proves the second route works and the response contract holds :
#        success flag , pattern string that PARSES as json (the system prompt
#        promises json - clients will json.loads it) , and the matched names
def test_get_pattern_returns_llm_pattern_json(monkeypatch, tmp_path):
    _install_happy_stubs(monkeypatch, tmp_path, _make_points("a.png"))

    response = client.post(
        "/get_pattern",
        data={"query": "what pattern is this?"},
        files={"context": ("upload.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["response"]["similar_images"] == ["a.png"]

    pattern = json.loads(body["response"]["pattern"])
    for key in ("summary", "style_tags", "dominant_colors", "recommendations"):
        assert key in pattern, f"llm pattern json missing key: {key}"


#what : the llm must receive exactly ONE image even when qdrant returns many points
#why   : ADDED - llama-3.2-vision rejects multi-image requests with a 400 (verified
#        live) . a regression in MAX_LLM_IMAGES would only explode at the nvidia
#        boundary in production , never in local tests without this capture check
def test_get_pattern_sends_only_top_match_to_llm(monkeypatch, tmp_path):
    _install_happy_stubs(monkeypatch, tmp_path, _make_points("a.png", "b.png", "c.png"))

    captured = {"images": None}

    def capturing_llm(query, image_content_list):
        captured["images"] = image_content_list
        return LLMResponse(response='{"summary": "s"}')

    monkeypatch.setattr(controller, "get_llm_response", capturing_llm)

    response = client.post(
        "/get_pattern",
        data={"query": "q"},
        files={"context": ("upload.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    assert captured["images"] is not None
    assert len(captured["images"]) == 1, "llm must receive at most 1 image (model limit)"


#what : /get_pattern stays 200 when some points cannot be resolved - one point has
#       no file_name in its payload , one download raises (like the live r2 404 on
#       test.jpeg) , one works
#why   : ADDED - partial bucket/db drift is the NORMAL case in production (we saw
#        a stale test.jpeg point live) . one bad point must degrade the result ,
#        not fail the whole request
def test_get_pattern_skips_unresolvable_points(monkeypatch, tmp_path):
    points = _make_points(None, "broken.png", "good.png")

    def fake_download(file_name, dest_folder):
        if file_name == "broken.png":
            raise Exception("404 Not Found from r2")
        dest = tmp_path / file_name
        dest.write_bytes(PNG_BYTES)
        return str(dest)

    _install_happy_stubs(monkeypatch, tmp_path, points)
    monkeypatch.setattr(controller, "download_files_from_s3", fake_download)

    response = client.post(
        "/get_pattern",
        data={"query": "q"},
        files={"context": ("upload.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["response"]["similar_images"] == ["good.png"]


#what : /get_similar_image zips only the files that actually downloaded - broken
#       points are skipped , the zip still returns 200 with the survivors
#why   : ADDED - same drift story for the zip path ; also guards the files_added
#        counter (an all-skipped request must NOT sneak through as 200)
def test_get_similar_image_zips_only_downloadable_files(monkeypatch, tmp_path):
    points = _make_points("broken.png", "good.png")

    def fake_download(file_name, dest_folder):
        if file_name == "broken.png":
            raise Exception("404 Not Found from r2")
        dest = tmp_path / file_name
        dest.write_bytes(PNG_BYTES)
        return str(dest)

    _install_happy_stubs(monkeypatch, tmp_path, points)
    monkeypatch.setattr(controller, "download_files_from_s3", fake_download)

    response = client.post(
        "/get_similar_image",
        files={"context": ("upload.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert zf.namelist() == ["good.png"]


# ════════════════════════════════════════════════════════════════════
# 2. NEGATIVE TESTS — these must fail in a specific way
# ════════════════════════════════════════════════════════════════════

#what : upload larger than 10 MB is rejected with 413 and names the limit
#why   : the size cap guards against memory blowups (base64 for the llm inflates
#        bytes ~33%) - it must reject BEFORE any downstream work runs
def test_get_pattern_oversize_413():
    oversize = PNG_BYTES + b"x" * (10 * 1024 * 1024)
    response = client.post(
        "/get_pattern",
        data={"query": "q"},
        files={"context": ("big.png", oversize, "image/png")},
    )
    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


#what : same size cap enforced on /get_similar_image
#why   : ADDED - both endpoints read the raw bytes into memory , the cap must not
#        exist on only one route
def test_get_similar_image_oversize_413():
    oversize = PNG_BYTES + b"x" * (10 * 1024 * 1024)
    response = client.post(
        "/get_similar_image",
        files={"context": ("big.png", oversize, "image/png")},
    )
    assert response.status_code == 413
 

#what : empty file body -> 400 "uploaded file is empty"
#why   : an empty upload is a client bug , not a missing field (that is 422) and
#        must be rejected before the embedding call
def test_get_pattern_empty_file_400():
    response = client.post(
        "/get_pattern",
        data={"query": "q"},
        files={"context": ("e.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "uploaded file is empty"


#what : non-image content type -> 400 naming the received type
#why   : this is an image pipeline - text/zip/video uploads must be rejected at
#        the door instead of producing garbage embeddings downstream
def test_get_pattern_non_image_content_type_400():
    response = client.post(
        "/get_pattern",
        data={"query": "q"},
        files={"context": ("f.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "text/plain" in response.json()["detail"]


#what : missing query field -> 422 from the fastapi validation layer
#why   : /get_pattern is a two-part contract (query + image) - a client that
#        forgets the query gets the standard validation error
def test_get_pattern_missing_query_422():
    response = client.post(
        "/get_pattern",
        files={"context": ("u.png", PNG_BYTES, "image/png")},
    )
    assert response.status_code == 422


#what : missing file part on both routes -> 422
#why   : the file is the core input ; forgetting it is a contract violation
def test_missing_context_file_422():
    r1 = client.post("/get_pattern", data={"query": "q"})
    r2 = client.post("/get_similar_image")
    assert r1.status_code == 422
    assert r2.status_code == 422


#what : form key named "request_context" (old name) instead of "context" -> 422
#why   : ADDED - this exact mismatch burned us twice in postman (Form-vs-File ,
#        field rename) . the test pins the client contract so a future param
#        rename breaks loudly here instead of silently in production clients
def test_wrong_field_name_422():
    response = client.post(
        "/get_similar_image",
        files={"request_context": ("u.png", PNG_BYTES, "image/png")},
    )
    assert response.status_code == 422


#what : embedding service fails (returns None) -> 502 "failed to create embedding"
#why   : CLIP being down is an upstream outage - it must surface as a clean 502 ,
#        never as a 500 crash or a None propagating into qdrant
def test_embedding_failure_502(monkeypatch, tmp_path):
    def failing_embed(data, filename):
        return None

    _install_happy_stubs(monkeypatch, tmp_path, _make_points("a.png"))
    monkeypatch.setattr(controller, "embed_image", failing_embed)

    response = client.post(
        "/get_pattern",
        data={"query": "q"},
        files={"context": ("u.png", PNG_BYTES, "image/png")},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "failed to create embedding for the uploaded image"


#what : qdrant fails (returns None) -> 503 "vector search is unavailable"
#why   : ADDED - a db outage is a different failure class than "nothing matched" ;
#        conflating them sends clients hunting for ghosts while infra is down
def test_vectordb_failure_503(monkeypatch, tmp_path):
    def failing_db(vector):
        return None

    _install_happy_stubs(monkeypatch, tmp_path, _make_points("a.png"))
    monkeypatch.setattr(controller, "get_embeddings_from_db", failing_db)

    response = client.post(
        "/get_pattern",
        data={"query": "q"},
        files={"context": ("u.png", PNG_BYTES, "image/png")},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "vector search is unavailable"


#what : db answers but with zero points -> 404 "no similar images found"
#why   : an empty index (or nothing within range) is a legitimate not-found , not
#        an error - the status distinction lets clients show "no matches" vs retry
def test_no_matches_404(monkeypatch, tmp_path):
    _install_happy_stubs(monkeypatch, tmp_path, _make_points())

    response = client.post(
        "/get_similar_image",
        files={"context": ("u.png", PNG_BYTES, "image/png")},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "no similar images found"


#what : matches exist but every download fails -> 502 "failed to load any of the
#       similar images" (not an empty zip , not a 200)
#why   : ADDED - returning an empty 200 zip would read as success to a client ;
#        total retrieval failure must be an explicit error
def test_all_downloads_failed_502(monkeypatch, tmp_path):
    def failing_download(file_name, dest_folder):
        raise Exception("r2 down")

    _install_happy_stubs(monkeypatch, tmp_path, _make_points("a.png", "b.png"))
    monkeypatch.setattr(controller, "download_files_from_s3", failing_download)

    r1 = client.post(
        "/get_similar_image",
        files={"context": ("u.png", PNG_BYTES, "image/png")},
    )
    r2 = client.post(
        "/get_pattern",
        data={"query": "q"},
        files={"context": ("u.png", PNG_BYTES, "image/png")},
    )
    assert r1.status_code == 502
    assert r1.json()["detail"] == "failed to load any of the similar images"
    assert r2.status_code == 502


#what : llm call fails (returns None) -> 502 "llm failed to generate a pattern"
#why   : ADDED - an nvidia outage (auth , quota , model pulled) must not 500 the
#        request after all the retrieval work already succeeded
def test_llm_failure_502(monkeypatch, tmp_path):
    def failing_llm(query, image_content_list):
        return None

    _install_happy_stubs(monkeypatch, tmp_path, _make_points("a.png"))
    monkeypatch.setattr(controller, "get_llm_response", failing_llm)

    response = client.post(
        "/get_pattern",
        data={"query": "q"},
        files={"context": ("u.png", PNG_BYTES, "image/png")},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "llm failed to generate a pattern"


# ════════════════════════════════════════════════════════════════════
# 3. SECURITY — tests that SHOULD be added (documented , not runnable yet)
#
#    the api does not have these protections today . each block names the test
#    to write and why , so the suite grows as soon as the guard lands .
# ════════════════════════════════════════════════════════════════════

# SECURITY-1 : no authentication on any route
# test_unauthenticated_request_gets_401 :
#     both endpoints are anonymous - anyone who can reach the server can burn
#     nvidia credits (each /get_pattern is a paid llm call) and pull arbitrary
#     files out of the r2 bucket through the retrieval path .
#     add an api-key header check (dependency or middleware) , then assert :
#       - no key      -> 401
#       - wrong key   -> 401
#       - valid key   -> 200

# SECURITY-2 : no rate limiting
# test_burst_requests_get_throttled :
#     without a per-client limit a single actor can hammer /get_pattern and
#     drain the nvidia quota / r2 egress budget . add slowapi or a gateway
#     limit , then assert N+1 rapid requests -> 429 .

# SECURITY-3 : content-type is trusted , magic bytes are not checked
# test_non_image_bytes_with_image_content_type_rejected :
#     validate_image_upload only inspects the client-controlled content-type
#     header . arbitrary bytes declared as image/png pass validation and get
#     forwarded to CLIP and (as base64) to nvidia - a data-exfil channel into
#     third-party services . sniff magic bytes (the sniffers already exist in
#     rag_utility) and reject mismatches with 400 before any egress .

# SECURITY-4 : path traversal via qdrant payload file_name
# test_traversal_file_name_stays_inside_embeds_dir :
#     file_name from the db payload flows unchecked into the s3 download path
#     (public/embeds/<file_name>) and the zip arcname . a poisoned point with
#     file_name "../../outside/evil.sh" makes download_files_from_s3 mkdir and
#     write outside public/embeds . anyone who can write to the index (or a
#     key collision in the bucket) controls that string . assert the written
#     path resolves inside public/embeds and the arcname has no directory
#     components .

# SECURITY-5 : size check runs AFTER the whole body is in memory
# test_huge_body_rejected_before_full_read :
#     await file_data.read() buffers the entire upload , then len() is checked .
#     a 2 GB body is fully in memory before the 413 fires - memory-exhaustion
#     DoS with a single request . enforce a content-length precheck (or stream
#     and abort) , then assert an oversized content-length gets 413 without
#     the server buffering it .

# SECURITY-6 : CORS wildcard combined with allow_credentials=True
# test_cors_config_is_safe :
#     allow_origins=["*"] + allow_credentials=True is an invalid/unsafe combo -
#     browsers reject it outright today , and if it ever "works" it means any
#     origin can make credentialed calls . pin explicit frontend origins once
#     known , then assert the cors headers echo only those origins .

# SECURITY-7 : unbounded zip accumulation on disk
# test_response_zips_are_cleaned_up :
#     every /get_similar_image writes similar_images_<hex>.zip under
#     public/embeds and nothing deletes them - repeated requests fill the disk
#     (local DoS) . use background cleanup or stream the zip , then assert the
#     file count in public/embeds does not grow per request .

# SECURITY-8 : secrets and account ids in logs
# test_error_logs_do_not_leak_secrets :
#     failure paths print full upstream error bodies (the nvidia 404 body
#     contains the account id) and the llm key sits in a module global .
#     assert captured log output contains no nvapi- prefixes and no account
#     ids before shipping logs anywhere persistent .
