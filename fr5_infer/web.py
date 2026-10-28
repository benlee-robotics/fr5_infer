"""Operator page: home / start / stop / reset, task, status, camera previews."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from fr5_infer.app import InferApp


WEB_DIR = Path(__file__).resolve().parent / "web"
APP_KEY = web.AppKey("infer", InferApp)


def make_web_app(infer: InferApp) -> web.Application:
    app = web.Application()
    app[APP_KEY] = infer
    app.router.add_get("/", index)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/home", command("HOME"))
    app.router.add_post("/api/stop", command("STOP"))
    app.router.add_post("/api/reset", command("RESET"))
    app.router.add_post("/api/start", start)
    app.router.add_post("/api/task", task)
    app.router.add_get("/camera/{name}.jpg", camera)
    return app


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB_DIR / "index.html")


async def status(request: web.Request) -> web.Response:
    return web.json_response(request.app[APP_KEY].status())


def command(name: str):
    async def handler(request: web.Request) -> web.Response:
        request.app[APP_KEY].send(name)
        return web.json_response({"ok": True, "command": name})

    return handler


async def start(request: web.Request) -> web.Response:
    return web.json_response(request.app[APP_KEY].start_episode())


async def task(request: web.Request) -> web.Response:
    body = await request.json()
    return web.json_response(request.app[APP_KEY].set_task(str(body.get("task", ""))))


async def camera(request: web.Request) -> web.Response:
    infer = request.app[APP_KEY]
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, infer.camera_jpeg, request.match_info["name"])
    if data is None:
        raise web.HTTPNotFound()
    return web.Response(body=data, content_type="image/jpeg", headers={"Cache-Control": "no-store"})
