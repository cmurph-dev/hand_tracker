
"""This is a hand tracking module that uses the MediaPipe library to detect and track hand 
landmarks in real-time using a webcam. It will open the camera and display the live feed.
The camera itself is flipped to mirror the user's movements, making it easier to interact 
with the hand tracking module. For example, raising your right hand will reflect on the
screen, with it being the right hand raised. The user can press the 'q' key to exit the
program."""

# Camera Stuff
import cv2 as cv
# Hand Tracking Stuff
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# Frames Per Second (FPS) Stuff
import time

ptime = 0

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

# Set up Hand Landmarker options
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE,
)

# Hand landmark connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

capture = cv.VideoCapture(0) # 0 is the default camera (webcam), 1 is the second camera (if available)

# Create the hand landmarker and process the video feed
with vision.HandLandmarker.create_from_options(options) as landmarker:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Mirror view
        frame = cv.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Create MediaPipe Image from the RGB frame
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        h, w, _ = frame.shape
            
        # Draw landmarks and connections
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                points = []

                # Convert normalized coords to pixel coords
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))

                    # Draw landmark
                    cv.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    cv.line(
                        frame,
                        points[start_idx],
                        points[end_idx],
                        (255, 0, 0),
                        2
                    )

        # Calculate FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        
        # Display FPS on the frame
        cv.putText(frame, 
            f'FPS: {int(fps)}',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv.imshow("Hand Tracking", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close windows
capture.release()
cv.destroyAllWindows()