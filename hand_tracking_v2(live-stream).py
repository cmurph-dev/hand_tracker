
"""
===========================================================
 Script Name : hand_tracking_v2(live-stream).py
 Description : This is a hand tracking module that uses the MediaPipe library to detect and track hand 
               landmarks in real-time using a camera. It will open the camera and display the live feed.
               Holding up a hand will show the landmarks and connections on the screen. For example, 
               raising your right hand will reflect on the screen.
 Author      : Christian Murph
 Created On  : 2026-05-19
 Last Modified: 2026-05-21
 Version     : 2.0.1
 Python Ver  : 3.13.13 (Microsoft Store)
 License     : ??? (What do I put here?)
===========================================================

Usage:
    1. Ensure you have the required dependencies installed (see Dependencies section).
    2. Run the script using Python 3.13.13 or later. (I guess?)
    3. The webcam will open, and you should see a live feed with hand tracking.
    4. Hold up your hand(s) in front of the camera to see the landmarks and connections.
    5. Press the 'q' key to exit the program.

Notes:
    - I actually have no idea how accurate the dependencies table is,
      I just put what I coded this in, but I have no idea if it will
      work on other versions.

Dependencies:
    - Python 3.13.13 (Microsoft Store)
    - MediaPipe 0.10.35
    - opencv-python 4.13.0.92

Change Log:
    v1.0   - Initial release (See hand_tracking_v1(image).py)
    v2.0   - Updated to use MediaPipe's live stream mode for improved performance and responsiveness.
    v2.0.1 - Minor code cleanup and documentation updates, alongside updated headers and comments for better readability.
             No functional changes were made in this version.
"""

# =========================
# Import required libraries
# =========================
import cv2 # OpenCV for video capture and display
import mediapipe as mp # MediaPipe for hand tracking
import time # Time for calculating FPS


# =========================
# Main Code Starts Here
# =========================
ptime = 0 
BaseOptions = mp.tasks.BaseOptions 
HandLandmarker = mp.tasks.vision.HandLandmarker 
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions 
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

# CALLBACK FUNCTION
def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result,
    num_hands=2
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

cap = cv2.VideoCapture(0) # 0 is the default camera (webcam), 1 is the second camera (if available)

# Create the hand landmarker and process the video feed
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Mirror View
        frame = cv2.flip(frame, 1)  

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image from the RGB frame
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame #this or data=frame works?
        )

        timestamp_ms = int(time.time() * 1000)

        # ASYNC DETECTION
        landmarker.detect_async(mp_image, timestamp_ms)

        # Draw hand landmarks and connections on the frame
        if latest_result and latest_result.hand_landmarks:
            h, w, _ = frame.shape
            
            # Draw landmarks and connections
            for hand_landmarks in latest_result.hand_landmarks:
                points = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)    
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 10, (0,0,255), -1)
            
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    cv2.line(
                        frame,
                        points[start_idx],
                        points[end_idx],
                        (255, 255, 255),
                        2
                    )

        # Calculate FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        
        # Display FPS on the frame
        cv2.putText(frame, 
            f'FPS: {int(fps)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display
        cv2.imshow("LIVE STREAM", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Kill the camera and close windows
cap.release()
cv2.destroyAllWindows()