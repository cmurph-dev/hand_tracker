import cv2
import mediapipe as mp
import time

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

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Mirror View
        frame = cv2.flip(frame, 1)  

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        timestamp_ms = int(time.time() * 1000)

        # ASYNC DETECTION
        landmarker.detect_async(mp_image, timestamp_ms)

        # DRAW RESULTS
        if latest_result and latest_result.hand_landmarks:

            h, w, _ = frame.shape

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

        cv2.imshow("LIVE STREAM", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Kill the camera and close windows
cap.release()
cv2.destroyAllWindows()