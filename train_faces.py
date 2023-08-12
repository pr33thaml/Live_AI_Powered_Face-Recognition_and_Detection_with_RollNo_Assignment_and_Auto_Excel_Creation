#do not forget to change the "logo png path"

import os
import face_recognition
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from PIL import Image, ImageTk

# Global variables to hold the PhotoImage references
logo_image = None
welcome_photo = None

def train():
    main_folder = main_folder_entry.get()
    if not main_folder:
        messagebox.showerror("Error", "Please select the training folder.")
        return

    train_button.config(state=tk.DISABLED)
    status_label.config(text="Training in progress...")

    def training_thread():
        try:
            training_folders = [os.path.join(main_folder, folder_name) for folder_name in os.listdir(main_folder) if
                                os.path.isdir(os.path.join(main_folder, folder_name))]

            known_face_encodings = []
            roll_numbers = []

            for training_folder in training_folders:
                roll_number = os.path.basename(training_folder)
                roll_numbers.append(roll_number)
                face_encodings = []
                for image_file in os.listdir(training_folder):
                    image_path = os.path.join(training_folder, image_file)
                    image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(image)[0]
                    face_encodings.append(face_encoding)
                known_face_encodings.append(face_encodings)

            data = {"encodings": known_face_encodings, "roll_numbers": roll_numbers}
            with open("encodings.pickle", "wb") as f:
                pickle.dump(data, f)

            status_label.config(text="Training successfully completed.")
            messagebox.showinfo("Success", "Training successfully completed.")
        except Exception as e:
            status_label.config(text="Training failed.")
            messagebox.showerror("Error", f"Training failed: {str(e)}")

        train_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=training_thread)
    thread.start()

# Create UI
root = tk.Tk()
root.title("Face Recognition Training")

# Set window icon
icon_image = Image.open(r"logo png path here")
icon_photo = ImageTk.PhotoImage(icon_image)
root.iconphoto(True, icon_photo)

# Set the original size of the window
original_width = 350
original_height = 120
window_geometry = f"{original_width}x{original_height}"
root.geometry(window_geometry)

# Load and display the logo image (in the window if desired)
# logo_label = tk.Label(root, image=logo_image)
# logo_label.pack()

training_folder_label = tk.Label(root, text="Select Training Folder:")
training_folder_label.pack()

main_folder_entry = tk.Entry(root)
main_folder_entry.pack()

browse_button = tk.Button(root, text="Browse", command=lambda: main_folder_entry.insert(tk.END, filedialog.askdirectory()))
browse_button.pack()

train_button = tk.Button(root, text="Train", command=train)
train_button.pack()

status_label = tk.Label(root, text="")
status_label.pack()

# Create welcome screen inside the window
def show_welcome_screen():
    global welcome_photo
    welcome_root = tk.Toplevel(root)
    welcome_root.title("Welcome!")

    welcome_image = Image.open(r"logo png path here")
    welcome_image = welcome_image.resize((400, 400))
    welcome_photo = ImageTk.PhotoImage(welcome_image)
    welcome_label = tk.Label(welcome_root, image=welcome_photo)
    welcome_label.pack()

    welcome_text = tk.Label(welcome_root, text="Welcome!", font=("Helvetica", 20))
    welcome_text.pack()

    # Bind a click event to close the welcome screen and show the main training screen
    welcome_label.bind("<Button-1>", lambda event, root=root: close_welcome_screen(welcome_root, root))

def close_welcome_screen(welcome_root, root):
    root.deiconify()  # Show the main training screen after the welcome screen is closed
    welcome_root.destroy()

# Hide the main training screen until the welcome screen is closed
root.withdraw()

# Show welcome screen
show_welcome_screen()

root.mainloop()
