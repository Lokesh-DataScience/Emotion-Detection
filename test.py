# Import necessary libraries
from keras.models import load_model
import cv2
import numpy as np

# Initialize the face classifier with the Haar Cascade model for face detection
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion classification model
# Uncomment the model you want to use and make sure the path is correct
classifier = load_model(r'Custom_CNN_model.keras')
# classifier = load_model(r'Final_Resnet50_Best_model.keras')

# Define the list of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start capturing video from the webcam (device 0 by default)
cap = cv2.VideoCapture(0)

# Continuous loop for live video feed
while True:
    # Read each frame from the video capture
    _, frame = cap.read()

    # Convert the frame to grayscale for the face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray)

    # Process each face detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Extract the region of interest (ROI) as the face area from the grayscale frame
        roi_gray = gray[y:y+h, x:x+w]
        # Resize the ROI to the size expected by the model (48x48 pixels in this case)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Proceed if the ROI is not empty
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0  # Normalize pixel values
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            # Predict the emotion of the face using the pre-trained model
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)

            # Display the predicted emotion label on the frame
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Display message if no faces are detected
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the detected faces and emotion labels
    cv2.imshow('Emotion Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
