# Emotion Detection from Video

A real-time emotion detection system that uses computer vision and deep learning to analyze facial expressions from video streams. This project detects emotions from faces captured via webcam and displays the detected emotion in real-time.

## 🎯 Features

- **Real-time emotion detection** from webcam feed
- **Face detection** using Haar Cascade classifier
- **Emotion classification** using DeepFace neural network
- **Live visualization** with emotion labels overlaid on video
- **Multiple emotion categories** including happy, sad, angry, surprised, neutral, and more
- **Error handling** for robust video processing

## 📋 Requirements

- Python 3.7+
- OpenCV (`cv2`)
- DeepFace
- NumPy
- Matplotlib

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rushiprasanthi/emotion-detection-from-video-updated-.git
   cd emotion-detection-from-video-updated-
   ```

2. **Install required dependencies:**
   ```bash
   pip install opencv-python deepface numpy matplotlib
   ```

3. **Additional requirements for DeepFace:**
   - DeepFace uses pre-trained models. The first run will download required model files automatically.

## 📁 Project Structure

```
emotion-detection-from-video-updated-/
├── README.md                              # This file
├── rushi11.py                             # Main emotion detection script
├── kanna.py                               # Utility script (matplotlib plots)
├── haarcascade_frontalface_default.xml    # Haar Cascade face detector
├── video1.mp4                             # Sample video file
└── __init__.py                            # Python package initialization
```

## 🚀 Usage

### Run Real-time Emotion Detection from Webcam

```bash
python rushi11.py
```

**Controls:**
- Press `Q` to exit the program
- The emotion detected will be displayed in green text on the video feed
- Console will print detected emotions in real-time

### How it Works

1. **Face Detection:** Uses Haar Cascade classifier to detect faces in each video frame
2. **Face Cropping:** Extracts the detected face region for accurate emotion analysis
3. **Emotion Analysis:** DeepFace analyzes the cropped face and predicts emotions
4. **Visualization:** Draws rectangles around detected faces and labels them with emotions
5. **Real-time Display:** Shows live video with emotion predictions

## 📊 Detected Emotions

The system can detect the following emotions:
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😲 Surprised
- 😐 Neutral
- 😨 Fear
- 🤢 Disgust

## ⚙️ Key Parameters

In `rushi11.py`, you can customize:

| Parameter | Description | Value |
|-----------|-------------|-------|
| `scaleFactor` | How much the image size is reduced at each image pyramid level | 1.1 |
| `minNeighbors` | How many neighbors each candidate rectangle should have to retain it | 5 |
| `enforce_detection` | Whether to enforce face detection before emotion analysis | False |

## 🎥 Sample Video Processing

To process a video file instead of webcam:

```python
# Change line 8 in rushi11.py from:
video = cv2.VideoCapture(0)

# To:
video = cv2.VideoCapture('video1.mp4')
```

## 📝 Code Example

```python
import cv2
from deepface import DeepFace

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Capture video
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Detect faces
    faces = face_cascade.detectMultiScale(frame, 1.1, 5)
    
    for (x, y, w, h) in faces:
        # Analyze emotion
        result = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        
        # Display on frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Emotion Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
```

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| No face detected | Ensure good lighting and that face is clearly visible |
| DeepFace model not found | First run will auto-download models (~350MB). Ensure internet connection |
| Webcam not opening | Check if another application is using the webcam, try `cv2.VideoCapture(1)` |
| Slow performance | Reduce frame resolution or adjust `scaleFactor` parameter |

## 🔍 Error Handling

The script includes error handling for:
- Failed frame capture
- Face detection failures
- Emotion analysis errors
- Invalid webcam access

## 📚 Dependencies Details

- **OpenCV**: Computer vision library for video processing
- **DeepFace**: Deep learning facial analysis library
- **NumPy**: Numerical computing library
- **Matplotlib**: Visualization library

## 🎓 Learn More

- [OpenCV Documentation](https://docs.opencv.org/)
- [DeepFace GitHub](https://github.com/serengp/deepface)
- [Haar Cascade Classifiers](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

## 💡 Future Enhancements

- [ ] Add emotion statistics and graphs
- [ ] Support multiple face detection and tracking
- [ ] Add emotion intensity levels
- [ ] Implement emotion timeline visualization
- [ ] Add video file processing with saved output
- [ ] Deploy as web application
- [ ] Add database for emotion history

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Rushi Prasanthi**

Feel free to contribute, report issues, or suggest improvements!

## 📧 Contact & Support

For questions or issues, please open a GitHub issue in this repository.

---

**Last Updated:** May 2026
