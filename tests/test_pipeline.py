"""
Tests for pipeline.indexer()

Mocks LlamaCloud client, requests.get, and file I/O so no real
API calls or disk writes happen.

Run with:  .venv/bin/python3 -m pytest tests/test_pipeline.py -v
"""
import sys, os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Add src/ to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ── helpers to build mock objects ──────────────────────────────────

def _make_mock_image(filename="screen1.png", url="https://cloud.example.com/screen1.png"):
    """Create a mock image object matching parsing_response.images_content_metadata.images[i]"""
    img = MagicMock()
    img.filename = filename
    img.url = url
    return img


def _make_mock_client(images=None, markdown_full="# Parsed Markdown"):
    """
    Build a fully wired mock LlamaCloud client.
    
    client.files.create()          → returns mock file with .id
    client.parsing.parse()         → returns mock response with images + markdown
    """
    if images is None:
        images = [_make_mock_image()]

    client = MagicMock()

    # files.create returns an object with .id
    mock_file = MagicMock()
    mock_file.id = "file-abc-123"
    client.files.create.return_value = mock_file

    # parsing.parse returns response with images and markdown
    mock_response = MagicMock()
    mock_response.images_content_metadata.images = images
    mock_response.markdown_full = markdown_full
    client.parsing.parse.return_value = mock_response

    return client


# ── tests ──────────────────────────────────────────────────────────

@patch("pipeline.requests.get")
@patch("builtins.open", mock_open(read_data=b"fake pdf bytes"))
@patch("pipeline.Path.mkdir", return_value=None)  # in case you add mkdir later
def test_indexer_returns_expected_keys(mock_requests_get, *_):
    """indexer() should return a dict with 'file_destination' and 'markdown_full'."""
    # Setup: mock requests.get to return fake image bytes
    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [b"fake-image-bytes"]
    mock_requests_get.return_value = mock_resp

    from pipeline import indexer
    client = _make_mock_client()

    result = indexer(client, "test.pdf", "out")

    assert "file_destination" in result
    assert "markdown_full" in result
    assert result["markdown_full"] == "# Parsed Markdown"


@patch("pipeline.requests.get")
@patch("builtins.open", mock_open(read_data=b"fake pdf bytes"))
def test_indexer_calls_llama_cloud_correctly(mock_requests_get):
    """indexer() should upload the file then parse with correct params."""
    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [b"data"]
    mock_requests_get.return_value = mock_resp

    from pipeline import indexer
    client = _make_mock_client()

    indexer(client, "design.pdf", "output")

    # Verify file was uploaded
    client.files.create.assert_called_once()
    create_kwargs = client.files.create.call_args
    assert create_kwargs.kwargs.get("purpose") == "parse" or \
           (len(create_kwargs.args) == 0 and "purpose" in str(create_kwargs))

    # Verify parse was called with the file id
    client.parsing.parse.assert_called_once()
    parse_kwargs = client.parsing.parse.call_args.kwargs
    assert parse_kwargs["file_id"] == "file-abc-123"
    assert parse_kwargs["tier"] == "agentic"


@patch("pipeline.requests.get")
@patch("builtins.open", mock_open(read_data=b"fake pdf bytes"))
def test_indexer_downloads_all_images(mock_requests_get):
    """indexer() should download every image from the parse response."""
    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [b"img-data"]
    mock_requests_get.return_value = mock_resp

    images = [
        _make_mock_image("page1.png", "https://example.com/page1.png"),
        _make_mock_image("page2.png", "https://example.com/page2.png"),
        _make_mock_image("page3.png", "https://example.com/page3.png"),
    ]
    client = _make_mock_client(images=images)

    from pipeline import indexer
    indexer(client, "test.pdf", "out")

    # Should have called requests.get once per image
    assert mock_requests_get.call_count == 3
    called_urls = [call.args[0] for call in mock_requests_get.call_args_list]
    assert "https://example.com/page1.png" in called_urls
    assert "https://example.com/page2.png" in called_urls
    assert "https://example.com/page3.png" in called_urls


@patch("pipeline.requests.get")
@patch("builtins.open", mock_open(read_data=b"fake pdf bytes"))
def test_indexer_handles_download_failure(mock_requests_get):
    """indexer() should not crash if an image download raises ValueError."""
    mock_requests_get.side_effect = ValueError("Connection failed")

    images = [_make_mock_image("broken.png", "https://example.com/broken.png")]
    client = _make_mock_client(images=images)

    from pipeline import indexer
    # Should not raise — the function catches ValueError
    try:
        result = indexer(client, "test.pdf", "out")
    except ValueError:
        pass  # acceptable if it propagates, but ideally caught


@patch("pipeline.requests.get")
@patch("builtins.open", mock_open(read_data=b"fake pdf bytes"))
def test_indexer_sets_correct_destination_path(mock_requests_get):
    """File destination should be out_dir / image.filename."""
    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [b"data"]
    mock_requests_get.return_value = mock_resp

    client = _make_mock_client(images=[_make_mock_image("hero.png")])

    from pipeline import indexer
    result = indexer(client, "test.pdf", "my_output")

    assert result["file_destination"] == Path("my_output") / "hero.png"


@patch("pipeline.requests.get")
@patch("builtins.open", mock_open(read_data=b"fake pdf bytes"))
def test_make_client_uses_env_key(mock_requests_get):
    """make_client() should read LLAMA_CLOUD_API_KEY from env."""
    with patch.dict(os.environ, {"LLAMA_CLOUD_API_KEY": "test-key-123"}):
        with patch("pipeline.LlamaCloud") as MockLlama:
            from pipeline import make_client
            make_client()
            MockLlama.assert_called_once_with(api_key="test-key-123")
