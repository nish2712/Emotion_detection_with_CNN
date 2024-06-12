import cv2
import numpy as np
from keras.models import model_from_json
import pygame

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the emotion recognition model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Start the webcam feed
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("C:/Users/Hp/Music/video.mp4")

# Initialize pygame for audio playback
pygame.mixer.init()  #loading and playing sound

# Create a dictionary mapping emotions to audio files
emotion_audio = {
    "Angry": "audio/angry_audio.mp3",
    "Disgusted": "audio/disgusted_audio.mp3",
    "Fearful": "audio/fearful_audio.mp3",
    "Happy": "audio/happy_audio.mp3",
    "Neutral": "audio/neutral_audio.mp3",
    "Sad": "audio/sad_audio.mp3",
    "Surprised": "audio/surprised_audio.mp3"
}

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720)) #resize an image frame to the dimensions 1280x720 pixels
    if not ret:
        break

    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        detected_emotion = emotion_dict[maxindex]

        cv2.putText(frame, detected_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Play audio corresponding to the detected emotion
        audio_file = emotion_audio.get(detected_emotion)
        if audio_file:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
