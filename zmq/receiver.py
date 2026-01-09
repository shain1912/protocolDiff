import asyncio
import cv2
import aiohttp
import supervision as sv
from aiortc import RTCPeerConnection, RTCSessionDescription
from ultralytics import YOLO
import time
import threading

# --- Supabase Stub (Mock) ---
class SupabaseStub:
    def __init__(self):
        print("[INFO] Supabase initialized (Mock Mode)")
    def table(self, name): return self
    def insert(self, data): return self
    def execute(self): pass

db = SupabaseStub()

async def run_receiver():
    model = YOLO("yolo11n.pt")
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    async with aiohttp.ClientSession() as session:
        async with session.post("http://127.0.0.1:8080/offer", json={
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type
        }) as resp:
            answer = await resp.json()
    
    await pc.setRemoteDescription(RTCSessionDescription(**answer))

    while True:
        receivers = pc.getReceivers()
        if receivers and receivers[0].track:
            remote_track = receivers[0].track
            break
        await asyncio.sleep(0.1)

    print("[INFO] WebRTC Receiver Started. Sampling at 5 FPS.")
    
    last_inference_time = 0
    inference_interval = 1.0 / 5.0 # 5 FPS

    try:
        while True:
            frame = await remote_track.recv()
            img = frame.to_ndarray(format="bgr24")

            # --- Sampling Logic ---
            current_time = time.time()
            if current_time - last_inference_time < inference_interval:
                cv2.imshow("WebRTC Receiver (PoC)", img)
                if cv2.waitKey(1) == ord('q'): break
                continue
            
            last_inference_time = current_time

            results = model(img)[0]
            detections = sv.Detections.from_ultralytics(results)

            if len(detections) > 0:
                 data = {
                    "timestamp": time.time(),
                    "object_count": len(detections),
                    "classes": detections.class_id.tolist()
                }
                 threading.Thread(target=lambda: db.table("detections").insert(data).execute()).start()

            img = box_annotator.annotate(img, detections)
            img = label_annotator.annotate(img, detections)
            
            cv2.circle(img, (30, 30), 10, (0, 0, 255), -1) 

            cv2.imshow("WebRTC Receiver (PoC)", img)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        await pc.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(run_receiver())
