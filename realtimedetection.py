import cv2
import numpy as np

from keras.models import Sequential
import json
with open('facialemotionmodel.json', 'r') as f:
    model_config = json.load(f)

model = Sequential.from_config(model_config['config'])  # ✅ safer for Keras 3.x


model.load_weights("facialemotionmodel.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Display the predicted emotion
            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            
        cv2.imshow("Output", im)
        cv2.waitKey(27)
        
    except cv2.error:
        pass
