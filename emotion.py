import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained emotion recognition model
model = load_model('emotion_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the input size of the model
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0

    # Reshape the image to match the model's input shape
    reshaped = normalized.reshape((1, 48, 48, 1))

    # Perform emotion prediction
    prediction = model.predict(reshaped)
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_labels[emotion_index]

    # Display the emotion label on the frame
    cv2.putText(frame, emotion_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Facial Emotion Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
