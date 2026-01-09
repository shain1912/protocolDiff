import argparse
import asyncio
import json
import logging
import os
import time
import numpy as np
import threading

import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

class CameraStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.last_time = time.time()
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # Start background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            try:
                success, frame = self.cap.read()
                if success:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            except Exception:
                pass

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        with self.lock:
            if self.frame is None:
                await asyncio.sleep(0.01)
                frame = np.zeros((480, 640, 3), np.uint8)
            else:
                frame = self.frame.copy()

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame
    
    def __del__(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    pc.addTrack(CameraStreamTrack())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )

pcs = set()

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    app = web.Application()
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)
    print("[INFO] WebRTC Sender running on http://0.0.0.0:8080")
    web.run_app(app, host="0.0.0.0", port=8080)
