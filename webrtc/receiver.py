import cv2
import zmq
import numpy as np
import time
import supervision as sv
from ultralytics import YOLO
import threading

# --- Supabase Stub (Mock) ---
class SupabaseStub:
    def __init__(self):
        print("[INFO] Supabase initialized (Mock Mode)")
    def table(self, name): return self
    def insert(self, data): return self
    def execute(self): pass

db = SupabaseStub()

def run_receiver():
    model = YOLO("yolo11n.pt")
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5555")
    socket.setsockopt(zmq.SUBSCRIBE, b'video')
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1) # Force keeping only latest message

    last_inference_time = 0
    inference_interval = 1.0 / 5.0 # 5 FPS

    print("[INFO] ZMQ Receiver Started. Sampling at 5 FPS.")
    
    try:
        while True:
            _, frame_data = socket.recv_multipart()
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            
            # --- Sampling Logic ---
            current_time = time.time()
            if current_time - last_inference_time < inference_interval:
                # Visualization only
                cv2.imshow('ZMQ Receiver (PoC)', frame)
                if cv2.waitKey(1) == ord('q'): break
                continue
                
            last_inference_time = current_time
            
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)

            if len(detections) > 0:
                 data = {
                    "timestamp": time.time(),
                    "object_count": len(detections),
                    "classes": detections.class_id.tolist()
                }
                 threading.Thread(target=lambda: db.table("detections").insert(data).execute()).start()

            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections)
            
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) 

            cv2.imshow('ZMQ Receiver (PoC)', frame)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        socket.close()
        context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_receiver()
