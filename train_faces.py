import os
import face_recognition
import pickle

# Step 1: Load Training Data
known_face_encodings = []
roll_numbers = []

training_folders = [
    r"C:\Users\Louis\Desktop\facerectraining\21BTRCL070",
    r"C:\Users\Louis\Desktop\facerectraining\21BTRCL079",  # Add new folder path
]

for training_folder in training_folders:
    for folder_name in os.listdir(training_folder):
        if os.path.isdir(os.path.join(training_folder, folder_name)):
            roll_number = folder_name
            roll_numbers.append(roll_number)
            folder_path = os.path.join(training_folder, folder_name)
            face_encodings = []
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)[0]
                face_encodings.append(face_encoding)
            known_face_encodings.append(face_encodings)

# Step 2: Save Encodings and Roll Numbers
data = {"encodings": known_face_encodings, "roll_numbers": roll_numbers}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("Training complete. Encodings saved.")
