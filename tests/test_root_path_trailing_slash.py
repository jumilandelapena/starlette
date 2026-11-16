"""Routing with a trailing-slash ``root_path`` under both ASGI server conventions.

Hypercorn, Daphne and Granian pass the request path through unchanged, so with
``root_path="/sub/"`` a request for ``/sub/who`` arrives as ``path="/sub/who"``.
uvicorn builds ``root_path + request_path``, so the same request arrives as
``path="/sub//who"``. Both must route, at every mount depth, from every entry
point, and the ``root_path`` an endpoint sees must stay a literal prefix of its
``path`` (the ASGI spec defines PATH_INFO as ``path`` minus ``root_path``).

Requests are sent as raw ASGI scopes rather than through TestClient, because an
HTTP client parses a leading ``//`` as a network-path reference.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from starlette._utils import get_route_path
from starlette.applications import Starlette
from starlette.routing import Mount, Route, Router
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Scope

SINGLE = "single"  # Hypercorn / Daphne / Granian: path passed through verbatim
DOUBLE = "double"  # uvicorn: path = root_path + request_path


def compose(root_path: str, request_path: str, convention: str) -> str:
    if convention == DOUBLE:
        return root_path + request_path
    return root_path.rstrip("/") + request_path


def call(app: ASGIApp, path: str, root_path: str) -> tuple[int, bytes]:
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": root_path,
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    sent: list[Message] = []
    requested = False

    async def receive() -> Message:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Event().wait()  # no disconnect until the app is done
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        sent.append(message)

    async def run() -> None:
        try:
            await app(scope, receive, send)
        except Exception as exc:  # bare Router / StaticFiles raise HTTPException
            status = getattr(exc, "status_code", 500)
            sent.append({"type": "http.response.start", "status": status})

    asyncio.run(run())
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], body


class Recorder:
    """Raw ASGI endpoint that records the scope it was called with."""

    def __init__(self) -> None:
        self.scopes: list[dict[str, Any]] = []

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        self.scopes.append(dict(scope))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def build(entry: str, depth: str, endpoint: Recorder) -> tuple[ASGIApp, str]:
    """Return (app, request path below root_path) for the endpoint."""
    if depth == "top":
        routes: list[Any] = [Route("/who", endpoint)]
        request = "/who"
    elif depth == "mount":
        routes = [Mount("/api", app=Router([Route("/who", endpoint)]))]
        request = "/api/who"
    else:  # nested
        inner = Router([Route("/who", endpoint)])
        routes = [Mount("/api", app=Router([Mount("/v1", app=inner)]))]
        request = "/api/v1/who"
    app: ASGIApp = Starlette(routes=routes) if entry == "starlette" else Router(routes)
    return app, request


@pytest.mark.parametrize("entry", ["starlette", "router"])
@pytest.mark.parametrize("depth", ["top", "mount", "nested"])
@pytest.mark.parametrize(
    "root_path,convention",
    [("/sub", SINGLE), ("/sub/", SINGLE), ("/sub/", DOUBLE)],
)
def test_routes_resolve(entry: str, depth: str, root_path: str, convention: str) -> None:
    endpoint = Recorder()
    app, request = build(entry, depth, endpoint)
    path = compose(root_path, request, convention)

    status, _ = call(app, path, root_path)

    assert status == 200, f"{path!r} with root_path={root_path!r} -> {status}"
    seen = endpoint.scopes[-1]
    assert seen["path"] == path
    assert seen["path"].startswith(seen["root_path"])
    assert get_route_path(seen) == "/who"


@pytest.mark.parametrize(
    "root_path,convention,expected_child_root",
    [
        ("/sub", SINGLE, "/sub/api"),
        ("/sub/", SINGLE, "/sub/api"),
        ("/sub/", DOUBLE, "/sub//api"),
    ],
)
def test_mount_child_root_path_is_consumed_prefix(root_path: str, convention: str, expected_child_root: str) -> None:
    endpoint = Recorder()
    app, request = build("router", "mount", endpoint)
    call(app, compose(root_path, request, convention), root_path)
    assert endpoint.scopes[-1]["root_path"] == expected_child_root


@pytest.mark.parametrize("root_path,convention", [("/sub/", SINGLE), ("/sub/", DOUBLE)])
@pytest.mark.parametrize("entry", ["bare", "mounted"])
def test_static_files(tmp_path: Path, root_path: str, convention: str, entry: str) -> None:
    (tmp_path / "a.txt").write_text("hello")
    static = StaticFiles(directory=tmp_path)
    if entry == "bare":
        app: ASGIApp = static
        request = "/a.txt"
    else:
        app = Starlette(routes=[Mount("/static", app=static)])
        request = "/static/a.txt"

    status, body = call(app, compose(root_path, request, convention), root_path)

    assert status == 200
    assert body == b"hello"


@pytest.mark.parametrize("convention", [SINGLE, DOUBLE])
def test_top_level_scope_not_mutated(convention: str) -> None:
    seen: list[dict[str, Any]] = []

    class Spy:
        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
            seen.append(dict(scope))
            await self.app(scope, receive, send)

    endpoint = Recorder()
    app = Starlette(routes=[Route("/who", endpoint)])
    path = compose("/sub/", "/who", convention)

    call(Spy(app), path, "/sub/")

    top = endpoint.scopes[-1]
    assert seen[0]["root_path"] == "/sub/"
    assert top["root_path"] == "/sub/"
    assert top["path"] == path
    assert top["raw_path"] == path.encode()


@pytest.mark.parametrize(
    "path,root_path,expected",
    [
        # trailing-slash root_path, single-slash path (Hypercorn/Daphne/Granian)
        ("/sub/who", "/sub/", "/who"),
        ("/sub/", "/sub/", "/"),
        ("/sub", "/sub/", ""),
        # trailing-slash root_path, uvicorn double slash
        ("/sub//who", "/sub/", "/who"),
        # root_path="/"
        ("/", "/", "/"),
        ("/a", "/", "/a"),
        ("//a", "/", "/a"),
        # unchanged behaviour
        ("/sub/who", "/sub", "/who"),
        ("/sub", "/sub", ""),
        ("/subway", "/sub", "/subway"),
        ("/subway", "/sub/", "/subway"),
        ("/sub-x/y", "/sub", "/sub-x/y"),
        ("/who", "", "/who"),
    ],
)
def test_get_route_path(path: str, root_path: str, expected: str) -> None:
    assert get_route_path({"path": path, "root_path": root_path}) == expected
