# Project: Real-time Face Detection
# Author: Hanzala Hussain Qureshi
# Date: April 2026
# Description: Detects faces in real-time using webcam and OpenCV Haar Cascade.

import cv2

# Load the pre-trained face detector model
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Turn on your webcam (0 = default webcam)
camera = cv2.VideoCapture(0)

while True:
    # Grab one frame from camera
    success, frame = camera.read()
    
    # Convert to grayscale (easier to detect patterns)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, 1.1, 5)
    
    # Draw a rectangle around each face and estimate distance
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        # Estimate distance using face width
        if w > 250:
            distance = "🔴 TOO CLOSE"
        elif w > 150:
            distance = "🟡 MEDIUM DISTANCE"
        else:
            distance = "🟢 FAR AWAY"
        
        # Display distance above the face box
        cv2.putText(frame,
                    distance,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)
    
    # Count faces and display
    face_count = len(faces)
    cv2.putText(frame,
                f'Faces detected: {face_count}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)
    
    # Show the result
    cv2.imshow('Mr. Ranedeer - Face Detector', frame)
    
    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
