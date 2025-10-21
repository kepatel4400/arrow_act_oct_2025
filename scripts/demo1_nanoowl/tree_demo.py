# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import argparse
from aiohttp import web, WSCloseCode
import logging
import weakref
import cv2
import time
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_predictor import OwlPredictor


def cv2_to_pil(image_bgr: np.ndarray) -> PIL.Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(rgb)


async def handle_index_get(request: web.Request):
    logging.info("handle_index_get")
    return web.FileResponse("./index.html")


def get_colors(count: int):
    cmap = plt.cm.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors


def build_app(args):
    app = web.Application()
    app["websockets"] = weakref.WeakSet()

    # predictor
    predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            image_encoder_engine=args.image_encode_engine
        )
    )
    app["predictor"] = predictor

    # shared state
    app["prompt_data"] = None  # dict with tree and encodings
    app["current_frame"] = {"img": None}  # last uploaded or captured frame
    app["image_quality"] = args.image_quality
    app["camera_device"] = args.camera
    app["width"], app["height"] = map(int, args.resolution.split("x"))

    async def websocket_handler(request: web.Request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        logging.info("Websocket connected")
        request.app["websockets"].add(ws)
        try:
            async for msg in ws:
                data = msg.data if isinstance(msg.data, str) else ""
                if "prompt" in data:
                    try:
                        _, prompt = data.split(":", 1)
                    except ValueError:
                        continue
                    prompt = prompt.strip()
                    logging.info("Received prompt: %s", prompt)
                    try:
                        tree = Tree.from_prompt(prompt)
                        clip_enc = predictor.encode_clip_text(tree)
                        owl_enc = predictor.encode_owl_text(tree)
                        request.app["prompt_data"] = {
                            "tree": tree,
                            "clip_encodings": clip_enc,
                            "owl_encodings": owl_enc,
                        }
                        logging.info("Prompt set")
                    except Exception as e:
                        logging.exception("Failed to set prompt: %s", e)
        finally:
            request.app["websockets"].discard(ws)
        return ws

    async def handle_upload(request: web.Request):
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file":
            return web.Response(status=400, text="expected form field 'file'")
        data = await field.read()
        npbuf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
        if img is None:
            return web.Response(status=400, text="could not decode image")
        w, h = request.app["width"], request.app["height"]
        if w > 0 and h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        request.app["current_frame"]["img"] = img
        return web.Response(text="ok")

    async def detection_loop(app: web.Application):
        loop = asyncio.get_running_loop()
        predictor = app["predictor"]
        prompt_data = lambda: app["prompt_data"]
        current_frame = app["current_frame"]
        image_quality = app["image_quality"]
        width, height = app["width"], app["height"]
        cam_index = app["camera_device"]

        camera = None
        if cam_index >= 0:
            logging.info("Opening camera %s", cam_index)
            camera = cv2.VideoCapture(cam_index)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        logging.info("Detection loop started")

        def _step():
            # acquire a frame
            if camera is not None:
                ok, frame = camera.read()
                if not ok:
                    return False, None
            else:
                src = current_frame["img"]
                if src is None:
                    time.sleep(0.05)
                    return True, None
                # work on a fresh copy every iteration so boxes do not accumulate
                frame = src.copy()

            # predict if prompt is set
            pd = prompt_data()
            if pd is not None:
                pil_img = cv2_to_pil(frame)
                t0 = time.perf_counter_ns()
                det = predictor.predict(
                    pil_img,
                    tree=pd["tree"],
                    clip_text_encodings=pd["clip_encodings"],
                    owl_text_encodings=pd["owl_encodings"],
                    threshold=0.3,
                )
                t1 = time.perf_counter_ns()
                _ = (t1 - t0) / 1e9  # dt seconds, reserved for logging if needed

                frame = draw_tree_output(frame, det, pd["tree"])
            # encode to jpeg
            enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, image_quality])[1]
            return True, bytes(enc)

        try:
            while True:
                ok, jpeg = await loop.run_in_executor(None, _step)
                if not ok:
                    break
                if jpeg is None:
                    continue
                for ws in list(app["websockets"]):
                    try:
                        await ws.send_bytes(jpeg)
                    except Exception:
                        pass
        finally:
            if camera is not None:
                camera.release()

    async def run_detection_loop(app):
        try:
            task = asyncio.create_task(detection_loop(app))
            yield
            task.cancel()
        except asyncio.CancelledError:
            pass
        finally:
            await task

    async def on_shutdown(app: web.Application):
        for ws in set(app["websockets"]):
            await ws.close(code=WSCloseCode.GOING_AWAY, message="Server shutdown")

    app.router.add_get("/", handle_index_get)
    app.router.add_post("/upload", handle_upload)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(run_detection_loop)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_encode_engine", type=str)
    parser.add_argument("--image_quality", type=int, default=50)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--camera", type=int, default=-1, help="set to -1 to disable camera")
    parser.add_argument("--resolution", type=str, default="640x480", help="WIDTHxHEIGHT")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    web.run_app(build_app(args), host=args.host, port=args.port)
