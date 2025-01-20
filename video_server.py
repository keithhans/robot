from flask import Flask, Response
import cv2
import threading
import time
import argparse

app = Flask(__name__)

# Global variables for video handling
video_capture = None
frame_lock = threading.Lock()
current_frame = None
stop_thread = False
camera_index = 0  # Default camera index

def init_camera():
    """Initialize the camera"""
    global video_capture
    video_capture = cv2.VideoCapture(camera_index)  # Use the specified camera index
    if not video_capture.isOpened():
        raise RuntimeError(f"Could not start camera with index {camera_index}")

def capture_frames():
    """Continuously capture frames from the camera"""
    global current_frame, stop_thread
    while not stop_thread:
        success, frame = video_capture.read()
        if success:
            with frame_lock:
                current_frame = frame.copy()
        time.sleep(0.03)  # Limit frame rate to ~30 fps

def generate_frames():
    """Generator function for streaming frames"""
    global current_frame
    while True:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                continue
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        # Convert to bytes and yield for streaming
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main page with video stream"""
    return """
    <html>
    <head>
        <title>Video Stream</title>
    </head>
    <body>
        <h1>Live Video Stream</h1>
        <img src="/video_feed" width="640" height="480" />
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Video streaming server')
        parser.add_argument('-c', '--camera', type=int, default=0,
                          help='Camera index (default: 0)')
        args = parser.parse_args()
        
        # Set camera index from command line argument
        camera_index = args.camera
        
        # Initialize camera
        init_camera()
        
        # Start frame capture thread
        capture_thread = threading.Thread(target=capture_frames)
        capture_thread.start()
        
        # Start Flask server
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        # Cleanup
        stop_thread = True
        if capture_thread.is_alive():
            capture_thread.join()
        if video_capture is not None:
            video_capture.release() 