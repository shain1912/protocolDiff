import cv2
import requests
import numpy as np
import time
import supervision as sv
from ultralytics import YOLO
import threading

# --- Supabase Stub (Mock) ---
# 실제 사용 시: from supabase import create_client, Client
class SupabaseStub:
    def __init__(self):
        print("[INFO] Supabase initialized (Mock Mode)")
        
    def table(self, name):
        return self
        
    def insert(self, data):
        # type: (dict) -> SupabaseStub
        # print(f"[DB LOG] Inserted into {data}")
        return self
        
    def execute(self):
        pass

# Initialize DB
db = SupabaseStub()
# db = create_client("YOUR_SUPABASE_URL", "YOUR_SUPABASE_KEY")

def run_receiver():
    model = YOLO("yolo11n.pt")
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    stream = requests.get("http://127.0.0.1:8001/video", stream=True)
    bytes_data = b''
    
    last_inference_time = 0
    inference_interval = 1.0 / 5.0  # 5 FPS (0.2s)

    print("[INFO] HTTP Receiver Started. Sampling at 5 FPS with Detection Persistence.")

    last_detections = None # Keep track of last results

    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            
            # Decode frame (Needed for visualization 30fps)
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            # --- Sampling Logic ---
            current_time = time.time()
            if current_time - last_inference_time >= inference_interval:
                # Run Inference (Only every 0.2s)
                last_inference_time = current_time
                
                results = model(frame)[0]
                detections = sv.Detections.from_ultralytics(results)
                last_detections = detections # Update "Memory"

                # --- Practice (Supabase) ---
                if len(detections) > 0:
                     data = {
                        "timestamp": time.time(),
                        "object_count": len(detections),
                        "classes": detections.class_id.tolist()
                    }
                     # Run in background to not block stream
                     threading.Thread(target=lambda: db.table("detections").insert(data).execute()).start()
            
            # --- Visualization (Always run 30fps using Memory) ---
            if last_detections is not None:
                frame = box_annotator.annotate(frame, last_detections)
                frame = label_annotator.annotate(frame, last_detections)
            
            # Draw indicators
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) 
            cv2.putText(frame, f"DB: {len(last_detections) if last_detections else 0} objs", (50, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('HTTP Receiver (PoC)', frame)
            if cv2.waitKey(1) == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_receiver()
