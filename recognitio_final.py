import face_recognition
import os
import cv2
import numpy as np
import math
from dlimage import download_image
import csv
import requests

# Helper function to calculate face confidence
def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []

    def __init__(self):
        pass

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.split('.')[0])
        print(self.known_face_names)

    def run_recognition(self, image_path):
        frame = cv2.imread(image_path)

        # Resize frame of image to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current image
        self.face_locations = face_recognition.face_locations(small_frame)
        self.face_encodings = face_recognition.face_encodings(small_frame, self.face_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = '???'

            # Calculate the shortest distance to face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])

            self.face_names.append(f'{name} ({confidence})')

        # Save face names to a CSV file
        output_csv = 'face_names.csv'  # Replace with the desired output CSV file path
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Face Number', 'Face Name'])
            for i, name in enumerate(self.face_names):
                writer.writerow([i+1, name])

        # Crop and save the detected faces
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            face_image = frame[top:bottom, left:right]

            face_filename = f'Detected_faces/{name.replace(" ", "_")}.jpg'

            cv2.imwrite(face_filename, face_image)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        resized_frame = cv2.resize(frame, (1000, 800))
        cv2.imshow('Face Recognition', resized_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.encode_faces()

    # Specify the URL of the JSON response on your PythonAnywhere web app
    json_url = 'https://mypresences.pythonanywhere.com/get_image_url'

    # Send a GET request to retrieve the JSON response
    response = requests.get(json_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        json_response = response.json()

        # Extract the image URL from the JSON response
        image_url = json_response['image_url']
        print(f"Image URL: {image_url}")

        # Download the image using the extracted URL
        image_response = requests.get(image_url)

        # Check if the image download request was successful (status code 200)
        if image_response.status_code == 200:
            # Save the image locally
            with open('image2.jpg', 'wb') as file:
                file.write(image_response.content)
            print('Image downloaded successfully.')
        else:
            print('Failed to download the image.')
    else:
        print('Failed to retrieve the JSON response.')

    image_path = 'image2.jpg'
    fr.run_recognition(image_path)
