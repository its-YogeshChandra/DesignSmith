"""
Integration tests for the real service functions behind the controller.

test_api.py monkeypatches every external call — these tests hit the REAL services
to verify the actual functions work end-to-end.

Services tested :
  1. CLIP   (localhost:8000) — free , local CPU
  2. Qdrant (localhost:6333) — free , local CPU
  3. R2     (Cloudflare)     — conservative , max 8-10 calls per run
  4. LLM    (NVIDIA NIM)     — 2-4 live calls max , vision model
  5. Pure utility functions  — no services needed

If a service is down , the tests FAIL (not skip). Fix the service and re-run.

Run with :  uv run pytest tests/test_integration.py -v
"""
import sys, os, struct, zlib, math, json
from pathlib import Path

# add src/ to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from designsmith.utils.rag_utility import (
    embed_image,
    embed_text,
    embed_images_batch,
    get_embeddings_from_db,
    get_llm_response,
    get_mimetype,
    bytes_to_data_uri,
    image_to_data_uri,
    LLMResponse,
)
from designsmith.utils.file_utility import download_files_from_s3, list_objects


# ── helpers ────────────────────────────────────────────────────────

def _make_png_bytes(r=0x11, g=0x22, b=0x33, width=8, height=8) -> bytes:
    """Build a tiny valid PNG with a single solid colour.
    Different r/g/b values produce visually distinct images for similarity tests."""
    raw_rows = b""
    for _ in range(height):
        raw_rows = raw_rows + b"\x00" + bytes([r, g, b]) * width

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBB B", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw_rows)) + _chunk(b"IEND", b""))


PNG_BYTES_A = _make_png_bytes(0x11, 0x22, 0x33)          # dark blue-ish
PNG_BYTES_A_COPY = _make_png_bytes(0x11, 0x22, 0x33)     # identical to A
PNG_BYTES_B = _make_png_bytes(0xFF, 0x00, 0x00)           # bright red — very different


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# known file that exists in both R2 and Qdrant (discovered from live data)
KNOWN_R2_FILE = "1.png"
# small file for LLM tests — large images cause NVIDIA NIM 500s on base64 payloads
SMALL_R2_FILE = "1 - A - Launch.png"


# ════════════════════════════════════════════════════════════════════
# GROUP 1 — CLIP embedding (local, free)
# ════════════════════════════════════════════════════════════════════

#what : embed raw bytes through the API upload path and verify the vector shape
#why   : monkeypatched tests never verify CLIP actually returns the right shape
def test_embed_image_from_bytes_returns_512d_vector():
    vector = embed_image(data=PNG_BYTES_A, filename="test.png")
    assert vector is not None, "CLIP returned None — is it running on localhost:8000?"
    assert isinstance(vector, list)
    assert len(vector) == 512
    for v in vector:
        assert math.isfinite(v), f"non-finite value in vector: {v}"


#what : embed via file path (used by the indexer pipeline)
#why   : exercises the file_path code branch which the bytes branch doesn't cover
def test_embed_image_from_file_returns_512d_vector(tmp_path):
    png_file = tmp_path / "test_file.png"
    png_file.write_bytes(PNG_BYTES_A)

    vector = embed_image(file_path=str(png_file))
    assert vector is not None, "CLIP returned None — is it running on localhost:8000?"
    assert len(vector) == 512


#what : embed a text string and verify the vector shape
#why   : embed_text is used for text-based search but has zero test coverage
def test_embed_text_returns_512d_vector():
    vector = embed_text("minimalist dashboard design")
    assert vector is not None, "CLIP text embedding returned None"
    assert len(vector) == 512


#what : send garbage bytes and verify graceful failure
#why   : verifies the error path — CLIP should reject it without crashing embed_image
def test_embed_image_invalid_data_returns_none():
    result = embed_image(data=b"not an image at all", filename="garbage.png")
    # either None (CLIP rejected it) or a vector (CLIP tried its best) — must not crash
    # the important thing is no unhandled exception
    assert result is None or (isinstance(result, list) and len(result) == 512)


#what : identical images have higher similarity than dissimilar ones
#why   : sanity-checks that CLIP embeddings are actually meaningful, not random floats
#        see docs/learning_stash.md for the full explanation
def test_two_similar_images_have_closer_vectors_than_dissimilar():
    vec_a = embed_image(data=PNG_BYTES_A, filename="a.png")
    vec_a_copy = embed_image(data=PNG_BYTES_A_COPY, filename="a_copy.png")
    vec_b = embed_image(data=PNG_BYTES_B, filename="b.png")

    assert vec_a is not None and vec_a_copy is not None and vec_b is not None

    sim_identical = _cosine_similarity(vec_a, vec_a_copy)
    sim_different = _cosine_similarity(vec_a, vec_b)

    assert sim_identical > sim_different, (
        f"identical pair similarity ({sim_identical:.4f}) should be greater than "
        f"dissimilar pair ({sim_different:.4f}) — CLIP may not be producing meaningful vectors"
    )


#what : batch embedding returns the right count and shape
#why   : batch embedding has zero test coverage — shape mismatch would silently corrupt the index
def test_embed_images_batch_returns_matching_count(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        p.write_bytes(_make_png_bytes(r=i * 80, g=0x44, b=0x88))
        paths.append(str(p))

    vectors = embed_images_batch(paths)
    assert len(vectors) == 3, f"expected 3 vectors, got {len(vectors)}"
    for v in vectors:
        assert len(v) == 512


# ════════════════════════════════════════════════════════════════════
# GROUP 2 — Qdrant vector DB (local, free)
# ════════════════════════════════════════════════════════════════════

#what : query with a real CLIP vector and verify the response structure
#why   : a Qdrant schema migration, renamed collection, or missing payload field
#        would pass all mocked tests but break production
def test_query_returns_query_response_with_points():
    # get a real vector from CLIP first
    vector = embed_image(data=PNG_BYTES_A, filename="query.png")
    assert vector is not None

    result = get_embeddings_from_db(vector)
    assert result is not None, "Qdrant returned None — is it running on localhost:6333?"
    assert hasattr(result, "points"), "result missing .points attribute"
    # collection has 176 points — we should get some back
    assert len(result.points) > 0, "query returned 0 points from a 176-point collection"
    # verify payload structure
    first = result.points[0]
    assert "file_name" in first.payload, "point payload missing 'file_name' key"


#what : pass None to the guard clause
#why   : tests the guard at line 231 against a real Qdrant connection (no mock)
def test_query_with_none_embedding_returns_none():
    result = get_embeddings_from_db(None)
    assert result is None


#what : random vector still returns a valid response (even if scores are low)
#why   : catches collection config issues (wrong vector size, deleted collection)
def test_query_with_random_vector_returns_points():
    import random
    random_vector = [random.uniform(-1.0, 1.0) for _ in range(512)]

    result = get_embeddings_from_db(random_vector)
    assert result is not None, "Qdrant returned None for a valid 512d vector"
    assert hasattr(result, "points")


# ════════════════════════════════════════════════════════════════════
# GROUP 3 — Cloudflare R2 (conservative, max 8-10 calls)
# ════════════════════════════════════════════════════════════════════

#what : list objects returns a non-empty list
#why   : zero test coverage — bad endpoint, expired creds, or wrong bucket only show up in prod
def test_list_objects_returns_list():
    objects = list_objects()
    assert isinstance(objects, list)
    assert len(objects) > 0, "R2 bucket is empty — expected indexed images"


#what : list objects entries have expected keys
#why   : the pipeline reads 'Key' from each entry — a schema change breaks indexing
def test_list_objects_entries_have_key():
    objects = list_objects()
    first = objects[0]
    assert "Key" in first, f"R2 object missing 'Key': {first}"
    assert "Size" in first, f"R2 object missing 'Size': {first}"


#what : download a known file and verify it lands on disk
#why   : the download function is the most failure-prone part (live R2 404 already happened)
def test_download_known_file_writes_to_disk(tmp_path):
    dest = str(tmp_path)
    result_path = download_files_from_s3(KNOWN_R2_FILE, dest)

    assert result_path is not None
    downloaded = Path(result_path)
    assert downloaded.exists(), f"file not found at {result_path}"
    assert downloaded.stat().st_size > 0, "downloaded file is empty"


#what : downloaded file is actually a valid image (starts with PNG/JPEG magic bytes)
#why   : a corrupted download would produce garbage embeddings without erroring
def test_download_known_file_is_valid_image(tmp_path):
    result_path = download_files_from_s3(KNOWN_R2_FILE, str(tmp_path))
    with open(result_path, "rb") as f:
        magic = f.read(8)

    is_png = magic.startswith(b"\x89PNG")
    is_jpeg = magic.startswith(b"\xff\xd8\xff")
    assert is_png or is_jpeg, f"downloaded file has unexpected magic bytes: {magic[:8]}"


#what : download a non-existent file raises an exception
#why   : a silent failure would produce corrupt zips — must raise
def test_download_nonexistent_file_raises(tmp_path):
    with pytest.raises(Exception):
        download_files_from_s3("definitely_not_a_real_file_xyz_999.png", str(tmp_path))


#what : download path is constructed correctly (no extra slashes, correct parent)
#why   : path bugs are subtle — a wrong dest means files land in unexpected places
def test_download_path_structure(tmp_path):
    result_path = download_files_from_s3(KNOWN_R2_FILE, str(tmp_path))
    assert result_path == str(tmp_path / KNOWN_R2_FILE)


#what : re-downloading the same file overwrites without error
#why   : concurrent requests or retries must not fail on existing files
def test_download_same_file_twice_no_error(tmp_path):
    dest = str(tmp_path)
    path_1 = download_files_from_s3(KNOWN_R2_FILE, dest)
    path_2 = download_files_from_s3(KNOWN_R2_FILE, dest)
    assert path_1 == path_2
    assert Path(path_2).exists()


# ════════════════════════════════════════════════════════════════════
# GROUP 4 — LLM (NVIDIA NIM, 2-4 live calls max)
# ════════════════════════════════════════════════════════════════════

#what : guard — no API key returns None without calling the API
#why   : tests the guard at line 283 without spending a cent
def test_llm_response_returns_none_without_api_key(monkeypatch):
    import designsmith.utils.rag_utility as rag
    monkeypatch.setattr(rag, "LLM_API_KEY", "")

    result = get_llm_response(
        query="test",
        image_content_list=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}}],
    )
    assert result is None


#what : guard — empty query returns None
#why   : tests the guard at line 286
def test_llm_response_returns_none_for_empty_query():
    result = get_llm_response(query="", image_content_list=[{"type": "text"}])
    assert result is None


#what : guard — empty image list returns None
#why   : tests the guard at line 289
def test_llm_response_returns_none_for_empty_images():
    result = get_llm_response(query="test query", image_content_list=[])
    assert result is None


#what : LIVE — send a real image to the vision LLM and verify the response structure
#why   : the only way to know the API key, model, and payload format actually work together
#cost  : 1 vision model call
def test_llm_live_response_returns_valid_json(tmp_path):
    # download a real image from R2 to build a proper data URI
    import base64
    image_path = download_files_from_s3(SMALL_R2_FILE, str(tmp_path))
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    mime = get_mimetype(SMALL_R2_FILE)
    image_content = [{
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }]

    result = get_llm_response(query="What design pattern is this?", image_content_list=image_content)

    assert result is not None, "LLM returned None — check API key and model availability"
    assert isinstance(result, LLMResponse)
    assert result.response is not None
    assert len(result.response.strip()) > 0, "LLM returned empty content"

    # the system prompt asks for JSON — verify it parses
    try:
        parsed = json.loads(result.response)
    except json.JSONDecodeError:
        pytest.fail(f"LLM response is not valid JSON: {result.response[:200]}")

    # verify expected keys from the system prompt contract
    for key in ("summary", "style_tags", "dominant_colors", "recommendations"):
        assert key in parsed, f"LLM JSON missing expected key: {key}"


#what : LIVE — verify the LLM handles a simple text query about the image
#why   : tests a different query style (short, direct) to catch model-specific quirks
#cost  : 1 vision model call
def test_llm_live_response_for_short_query(tmp_path):
    import base64
    image_path = download_files_from_s3(SMALL_R2_FILE, str(tmp_path))
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    mime = get_mimetype(SMALL_R2_FILE)
    image_content = [{
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }]

    result = get_llm_response(query="describe the colors", image_content_list=image_content)

    assert result is not None, "LLM returned None for a simple query"
    assert len(result.response.strip()) > 0


#what : LLMResponse class stores text correctly
#why   : pins the contract — a refactor to pydantic model or dataclass would be caught
def test_llm_response_class_stores_text():
    lr = LLMResponse(response="hello world")
    assert lr.response == "hello world"


# ════════════════════════════════════════════════════════════════════
# GROUP 5 — Pure utility functions (no service calls)
# ════════════════════════════════════════════════════════════════════

#what : get_mimetype returns correct types for common extensions
#why   : used in data URI construction — wrong mime means a broken data: prefix
def test_get_mimetype_common_extensions():
    assert get_mimetype("photo.png") == "image/png"
    assert get_mimetype("photo.jpg") == "image/jpeg"
    assert get_mimetype("photo.jpeg") == "image/jpeg"
    assert get_mimetype("icon.svg") == "image/svg+xml"
    assert get_mimetype("photo.webp") == "image/webp"
    # unknown extension falls back to image/jpeg
    assert get_mimetype("file.qqqxyz") == "image/jpeg"


#what : bytes_to_data_uri builds a valid data URI that round-trips
#why   : the data URI is the literal input to CLIP — malformed = garbage embeddings
def test_bytes_to_data_uri_format():
    import base64
    mime, uri = bytes_to_data_uri(PNG_BYTES_A, "test.png")
    assert mime == "image/png"
    assert uri.startswith("data:image/png;base64,")
    # round-trip : decode the base64 back and compare
    b64_part = uri.split(",", 1)[1]
    decoded = base64.b64decode(b64_part)
    assert decoded == PNG_BYTES_A


#what : image_to_data_uri reads a file and builds a valid data URI
#why   : exercises the file-reading path used by the indexer
def test_image_to_data_uri_reads_file(tmp_path):
    import base64
    png_file = tmp_path / "test.png"
    png_file.write_bytes(PNG_BYTES_A)

    mime, uri = image_to_data_uri(str(png_file))
    assert mime == "image/png"
    assert uri.startswith("data:image/png;base64,")
    b64_part = uri.split(",", 1)[1]
    assert base64.b64decode(b64_part) == PNG_BYTES_A
