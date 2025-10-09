#!/usr/bin/env python3
import os
import PIL
import socket
import fastapi
import uvicorn
import logging
import threading

import numpy as np
import gradio as gr

from fastapi.staticfiles import StaticFiles


class Server(threading.Thread):
    def __init__(self, db, host='0.0.0.0', port=7860):
        super(Server, self).__init__(daemon=True)  # stop thread on main() exit

        self.db = db
        self.app = fastapi.FastAPI()
        self.host = host
        self.port = port
        self.mounts = {}
        self.server_url = f"http://{socket.gethostname()}:{port}"

        self.gallery_size = 64
        self.gallery_images = None

        # mount each scan root for static serving
        for n, scan in enumerate(db.scans):
            self.mounts[scan] = f"/files/{n}/"
            self.app.mount(self.mounts[scan], StaticFiles(directory=scan), name=str(n))

        self.create_ui()

    def run(self):
        logging.info(f"starting nanodb webserver at {self.server_url}")
        uvicorn.run(self.app, host=self.host, port=self.port, reload=False, log_level='warning')

    def get_random_images(self, n):
        indexes = np.random.randint(0, len(self.db) - 1, n)
        images = []
        for m in range(n):
            path = self.db.metadata[indexes[m]]['path']
            images.append(path)
        return images

    def create_ui(self):
        css = """
            #stats_box {font-family: monospace; font-size: 65%; height: 162px;}
            footer {visibility: hidden}
            body {overflow: hidden;}
        """
        with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as blocks:
            gr.HTML('<h1 style="color: #6aa84f; font-size: 250%;">nanodb</h1>')

            with gr.Row().style(equal_height=True):
                with gr.Column():
                    text_query = gr.Textbox(placeholder="Search Query", show_label=False)
                    stats = gr.Textbox(
                        value=self.create_stats(),      # fixed: call without passing built-in 'type'
                        lines=5,
                        show_label=False,
                        interactive=False,
                        elem_id='stats_box'
                    )
                # image upload removed (text-only UI)

            self.gallery_images = self.get_random_images(self.gallery_size)

            gallery = gr.Gallery(
                value=self.gallery_images
            ).style(columns=8, height='750px', object_fit='scale_down', preview=False)

            # callbacks without image widget
            gallery.select(self.on_gallery_select, None, [gallery, stats, text_query], show_progress=False)
            text_query.change(self.on_query, text_query, [gallery, stats], show_progress=False)

        self.app = gr.mount_gradio_app(self.app, blocks, path='/')

    def create_stats(self, type=None):
        text = f"Model:   CLIP {self.db.model.config.name}\n"
        text += f"Images:  {len(self.db):,}\n\n"

        if type == 'image' and 'time' in self.db.model.vision.stats:
            text += f"Image Encode:  {self.db.model.vision.stats.time*1000:3.1f} ms\n"

        if type == 'text' and 'time' in self.db.model.text.stats:
            text += f"Text Encode:   {self.db.model.text.stats.time*1000:3.1f} ms\n"

        if 'search_time' in self.db.index.stats:
            text += f"KNN Search:    {self.db.index.stats.search_time*1000:3.1f} ms"

        return text

    def on_query(self, query):
        """
        Text-only user input. Accepts:
          - str (free text or image path)
        Note: gallery clicks still pass an image path (str), which we treat as an image query.
        """
        if query is None:
            return gr.Gallery.update(value=self.gallery_images), gr.Textbox.update(value=self.create_stats())

        if isinstance(query, str):
            if os.path.splitext(query)[1].lower() in self.db.img_extensions:
                logging.debug(f"image query from path {query}")
                query_type = 'image'
            else:
                logging.debug(f"web text query '{query}'")
                query_type = 'text'
        else:
            raise ValueError(f"unexpected query type {type(query)}")

        indexes, distances = self.db.search(query, k=self.gallery_size)
        images = []
        for n in range(self.gallery_size):
            images.append((self.db.metadata[indexes[n]]['path'], f"{distances[n]*100:.1f}%"))

        self.gallery_images = images
        return gr.Gallery.update(value=images), gr.Textbox.update(value=self.create_stats(query_type))

    def on_gallery_select(self, evt: gr.SelectData):
        logging.debug(f"web client selected {evt.value} at {evt.index} from {evt.target}  selected={evt.selected}")
        if evt.index < len(self.gallery_images):
            img = self.gallery_images[evt.index]
            if not isinstance(img, str):
                img = img[0]
        else:
            img = "/data/images/lake.jpg"

        gal_update, stats = self.on_query(img)
        # third output (text_query) we clear to keep the box empty
        return gal_update, stats, ""
