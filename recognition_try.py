import face_recognition
import os
import cv2
import numpy as np
import math
from dlimage import download_image
import csv

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

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        self.face_locations = face_recognition.face_locations(small_frame)
        self.face_encodings = face_recognition.face_encodings(small_frame, self.face_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            self.face_names.append(name)

        output_csv = 'face_names.csv'
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Face Number', 'Face Name'])
            for i, name in enumerate(self.face_names):
                writer.writerow([i+1, name])

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

            # Remove the percentage from the displayed name
            display_name = name.split(' ')[0]

            cv2.putText(frame, display_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        resized_frame = cv2.resize(frame, (1000, 800))
        cv2.imshow('Face Recognition', resized_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_recognition(img_url):
    fr = FaceRecognition()
    fr.encode_faces()
    image_url = "https://i.imgur.com/a0aBCEw.jpg"
    download_image(image_url, "image1.jpg")
    image_path = 'image.jpg'
    fr.run_recognition(image_path)
