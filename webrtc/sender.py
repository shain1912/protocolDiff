import cv2
import zmq
import time
import threading

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        
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
            if self.frame is None: return None
            return self.frame.copy()

def run_sender():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # HWM=1 to drop old frames if receiver is slow
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind("tcp://*:5555")
    
    camera = VideoCamera()
    print("[INFO] ZMQ PoC Sender started on tcp://*:5555")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame)
            
            try:
                socket.send(b'video', zmq.SNDMORE)
                socket.send(buffer)
            except zmq.Again:
                pass 
            
            # 30 FPS Emission
            time.sleep(0.033)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        context.term()

if __name__ == "__main__":
    run_sender()
