import cv2
from deepface import DeepFace

# Load the face detector
face_values = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open video capture (0 for webcam)
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    
    if not ret or frame is None:
        print("Failed to capture frame")
        continue  # Skip processing if no frame

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_values.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        
        try:
            # Crop the detected face for better accuracy
            face_crop = frame[y:y + h, x:x + w]

            # Analyze emotions
            analyze = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)

            # Print detected emotion
            dominant_emotion = analyze[0]['dominant_emotion']
            print("Emotion:", dominant_emotion)

            # Display emotion on the video
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print("Emotion detection error:", str(e))

    # Show video feed
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit
    key = cv2.waitKey(1)
    if key == ord("q"):  
        break

# Release video and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

