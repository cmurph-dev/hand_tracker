# Header!
"""
===========================================================
 Script Name : hand_tracking.py
 Description : This is a hand tracking module that uses the MediaPipe library to detect and track hand 
               landmarks in real-time using a camera. It will open the camera and display the live feed.
 Author      : Christian Murph
 Created On  : 2026-05-19
 Last Modified: 2026-05-22
 Version     : 2.1.0
 Python Ver  : 3.13.13 (Microsoft Store)
 License     : ??? (What do I put here?)
===========================================================

Usage:
    1. Ensure you have the required dependencies installed (see Dependencies section).
    2. Run the script using Python 3.13.13 or later. (I guess?)
    3. Modify settings such as num_hands, min_hand_detection_confidence, min_tracking_confidence, and
     min_hand_presence_confidence in the code to adjust the hand tracking performance as needed.
    4. The webcam will open, and you should see a live feed with hand tracking.
    5. Hold up your hand(s) in front of the camera to see the landmarks and connections.
    6. Press the 'q' key to exit the program.

Notes:
    - I actually have no idea how accurate the dependencies table is, I just put what I coded this in, but
     I have no idea if it will work on other versions.
    - Do Devs actually put notes here? Why? There's a change log lol.

Dependencies:
    - Python 3.13.13 (Microsoft Store)
    - MediaPipe 0.10.35
    - opencv-python 4.13.0.92

Change Log:
    v1.0   - Initial release with vision running mode set to IMAGE, allowing for single image processing.
    v2.0   - Updated to use MediaPipe's live stream mode for improved performance and responsiveness.
    v2.0.1 - Minor code cleanup and documentation updates, alongside updated headers and comments for 
     better readability. No functional changes were made in this version.
    v2.0.2 - Made code look readable/prettier. Added  settings to play around with. Updated headers and
     comments for better readability. No functional changes were made in this version.
    v2.1.0 - Threading! Improved performance by using a separate thread for video capture, allowing the 
     main thread to focus on processing and displaying results, thereby improving overall performance. 
     Updated headers and comments for better readability. No functional changes were made in this version.
"""

# =========================
# Import required libraries
# =========================
import cv2 # OpenCV for video capture and display
import mediapipe as mp # MediaPipe for hand tracking
import time # Time for calculating FPS
import threading # Threading for asynchronous processing


# =========================
# Main Code Starts Here
# =========================


# Initialize MediaPipe hand landmarker and other variables
ptime = 0 
BaseOptions = mp.tasks.BaseOptions 
HandLandmarker = mp.tasks.vision.HandLandmarker 
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions 
VisionRunningMode = mp.tasks.vision.RunningMode
latest_result = None
latest_frame = None
running = True
timestamp_ms = 0

# CALLBACK FUNCTION
def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = on_result,
    num_hands = 2,
    min_hand_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    min_hand_presence_confidence = 0.5
)
HAND_CONNECTIONS = [
    (0, 5), (5,9), (9,13), (13,17), (0,17), # Palm connections
    (0, 1), (1,2), (2,3), (3,4), # Thumb connections
    (5, 6), (6,7), (7,8), # Index finger connections
    (9, 10), (10,11), (11,12), # Middle finger connections
    (13, 14), (14,15), (15,16), # Ring finger connections
    (17, 18), (18,19), (19,20)  # Pinky finger connections
]

cap = cv2.VideoCapture(0) # 0 is the default camera (webcam), 1 is the second camera (if available)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce buffer size to minimize latency
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480) # Set the width of the video feed to (pixels)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Set the height of the video feed (pixels)

# Thread for video capture
def capture_loop():
    global latest_frame
    while running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  
            latest_frame = frame
capture_thread = threading.Thread(target=capture_loop, daemon=True)


# Create the hand landmarker and process the video feed
with HandLandmarker.create_from_options(options) as landmarker:
     capture_thread.start()

     # Main loop to process video frames and display results
     while True:
         if latest_frame is None: 
             continue 
         frame = latest_frame.copy() 
         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) 
         timestamp_ms += 1 
         landmarker.detect_async(mp_image, timestamp_ms) 

          # Draw hand landmarks and connections on the frame
         if latest_result and latest_result.hand_landmarks:
              for hand_landmarks in latest_result.hand_landmarks:
                  points = []
                  h, w, _ = frame.shape
                  for landmark in hand_landmarks:
                      x = int(landmark.x * w)
                      y = int(landmark.y * h)    
                      points.append((x, y))
                      cv2.circle(frame, (x, y), 10, (0,0,255), -1)
                  for connection in HAND_CONNECTIONS:
                      cv2.line(
                          frame,
                          points[connection[0]],
                          points[connection[1]],
                          (255, 255, 255),
                          2
                      )

         # Calculate & Display FPS
         ctime = time.time()
         fps = 1 / (ctime - ptime)
         ptime = ctime
         cv2.putText(frame, 
             f'FPS: {int(fps)}',
             (10, 30),
             cv2.FONT_HERSHEY_SIMPLEX,
             1,
             (0, 255, 0),
             2
         )

         # Display Everything!
         cv2.imshow("LIVE STREAM", frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
             running = False
             break

capture_thread.join()
# Kill the camera and close windows
cap.release()
cv2.destroyAllWindows()