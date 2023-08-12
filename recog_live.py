import face_recognition
import pickle
from openpyxl import Workbook
import cv2
import datetime

with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)
known_face_encodings = data["encodings"]
roll_numbers = data["roll_numbers"]

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize data storage
detected_faces = []  # List to temporarily store detected faces and statuses
recorded_roll_numbers = set()  # Set to store recorded roll numbers
present_count = 0  # Initialize present count

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        roll_number_match = "Unknown"
        status = "Absent"  # Default status

        for i, known_encoding_list in enumerate(known_face_encodings):
            matches = face_recognition.compare_faces(known_encoding_list, face_encoding)
            if any(matches):
                roll_number_match = roll_numbers[i]
                status = "Present"  # Change status to "Present"
                break

        if roll_number_match != "Unknown":
            if roll_number_match not in recorded_roll_numbers:
                detected_faces.append((roll_number_match, status))
                recorded_roll_numbers.add(roll_number_match)
                present_count += 1  # Increase present count for new face

    # Display present count on video frame
    font = cv2.FONT_HERSHEY_DUPLEX
    present_text = f"Present: {present_count}"
    cv2.putText(frame, present_text, (10, 30), font, 1.0, (0, 255, 0), 1)  # Set color to green

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Close the camera window
cap.release()
cv2.destroyAllWindows()

# Sort detected_faces by roll number before saving
detected_faces.sort(key=lambda x: x[0])

# Save the detected faces and statuses to Excel
excel_file_path = "detected_faces.xlsx"
wb = Workbook()
ws = wb.active
ws.title = "Detected Faces"
ws.append(["Roll Number", "Status"])
for face_info in detected_faces:
    if face_info[0] != "Unknown":
        ws.append(face_info)
wb.save(excel_file_path)

print(f"Recognition complete. Detected faces saved to {excel_file_path}")
