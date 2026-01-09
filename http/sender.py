import cv2
import threading
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        
        if not self.video.isOpened():
            print("[ERROR] Could not open camera.")
            
        # Start background frame reading
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __del__(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()

    def update(self):
        while self.running:
            success, frame = self.video.read()
            if success:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)
                
    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

# Global camera instance
camera_stream = VideoCamera()

def generate_frames():
    while True:
        frame = camera_stream.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
            
        # JPEG Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Output MJPEG frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Stream at ~30 FPS (Server sends everything)
        time.sleep(0.033)

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("[INFO] Starting HTTP PoC Sender...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
