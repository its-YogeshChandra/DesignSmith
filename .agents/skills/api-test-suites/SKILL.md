---
name: api-test-suites
description: Write API test suites split into positive tests, negative tests and security-flaws sections. Use whenever the user asks to write, extend or review tests for an HTTP API or backend endpoint, mentions positive/negative/security tests, or asks for a test suite file — even if they only say "add tests for this endpoint".
---

# API test suites (positive / negative / security)

Every test suite written under this skill has exactly three sections, in this
order: positive tests (must pass), negative tests (must fail a specific way),
and security flaws (documented gaps). The suite tests real application code —
only external services are faked.

## File layout

One file per api surface (e.g. `tests/test_api.py`), structured:

1. Module docstring: what is covered, what is mocked, the run command.
2. Helpers (`_`-prefixed plain functions — no lambda logic).
3. `# 1. POSITIVE TESTS` section.
4. `# 2. NEGATIVE TESTS` section.
5. `# 3. SECURITY` section.

Use divider comment lines between sections so the three groups are greppable.

## Every test carries a what/why block

Above each test, in this exact shape:

```python
#what : one line what the test does
#why   : why this behaviour must hold
def test_something():
    ...
```

When you add a test the user did not ask for, mark it inside the why with
`ADDED - <reason>`. The user must be able to tell their tests from yours.

## Section 1 — positive tests

- Happy paths through the real routing layer (TestClient / equivalent), not
  direct function calls — the route wiring, field names and validation are
  part of what must keep working.
- Assert the response contract, not just 200: content type, body shape,
  file names inside a zip, keys of a json payload, parsed values.
- If the endpoint promises a format (e.g. "llm returns json"), assert the
  promise holds at the api boundary (json.loads succeeds, keys exist).
- Add tests for gaps you notice (partial upstream failures, caps, limits,
  capture-what-was-forwarded checks) with an `ADDED -` reason. Prefer one
  test per behaviour.

## Section 2 — negative tests

Each test must fail a *specific* way: assert the exact status code AND the
detail message. Status taxonomy to apply consistently:

| status | meaning                                                        |
|-------|----------------------------------------------------------------|
| 400   | client sent something malformed (empty file, wrong content type) |
| 404   | legitimate not-found (empty result set)                         |
| 413   | request exceeds a size limit                                    |
| 422   | request violates the field contract (missing/renamed fields)    |
| 429   | rate limited                                                    |
| 502   | an upstream dependency failed (embedding, llm, storage)         |
| 503   | infrastructure unavailable (db down)                            |

- Keep failure classes distinct: "db down" (503) is not "nothing matched"
  (404); conflating them lies to clients.
- Cover every guard the endpoint has: size caps on every route, empty
  payloads, wrong content types, missing/renamed form fields, each external
  dependency returning failure.
- An endpoint that returns success while doing nothing (empty zip, empty
  list after total retrieval failure) is a bug — pin it as a negative test.

## Section 3 — security flaws

When a protection does not exist yet, do NOT write a failing test — write a
commented block per gap, each shaped:

```
# SECURITY-<n> : <one-line flaw>
# test_<suggested_test_name> :
#     why it matters (concrete abuse path) .
#     add <the guard to build> , then assert :
#       - <exact expected behaviour>
```

Checklist of areas to walk through before finishing: authentication,
rate limiting, input content actually validated (magic bytes, not client
headers), path traversal through stored metadata (payloads, keys used in
filesystem paths or archive entries), checks that run after full buffering
(memory DoS), CORS configuration, unbounded on-disk artifact accumulation,
secrets or account ids leaking into logs.

## Isolation rules

- Never hit real external services (paid llms, vector dbs, object storage,
  embedding apis). Stub them per-test.
- Stub the name where it is *used*, not where it is defined: if the
  controller does `from x import embed`, monkeypatch `controller.embed` —
  patching the origin module does nothing.
- Stubs that stand in for "download a file" must write a real temp file when
  the code under test opens the returned path.
- Build structurally valid payloads with the stdlib (a real tiny png, valid
  json strings) — the code under test may parse them.
- If the user says they will run the tests themselves, only compile-check
  (`python -m py_compile`) and say so — never execute the suite.

## Skeleton

```python
"""
Tests for <routes> : ...
External services (<list>) are never hit - <which names> are monkeypatched.
Run with:  uv run pytest tests/test_api.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fastapi.testclient import TestClient
from designsmith.server import app
import controllers.user_controller as controller

client = TestClient(app, raise_server_exceptions=False)

# ── helpers ──
def _make_points(*file_names): ...

# ══ 1. POSITIVE TESTS ══
#what : ...
#why   : ...
def test_happy_path(monkeypatch, tmp_path): ...

# ══ 2. NEGATIVE TESTS ══
#what : ...
#why   : ...
def test_oversize_413(): ...

# ══ 3. SECURITY — documented gaps ══
# SECURITY-1 : no authentication on any route
# test_unauthenticated_request_gets_401 : ...
```
