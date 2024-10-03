import os
import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
import pickle

# Known face encodings and names
known_face_encodings = []
known_face_names = []

# Function to save known faces
def save_known_faces():
    with open('known_faces.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

# Function to load known faces
def load_known_faces():
    global known_face_encodings, known_face_names
    try:
        with open('known_faces.pkl', 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
    except FileNotFoundError:
        st.warning("No known faces found, please add some.")

# Helper function to add new face
def add_new_face(image, name):
    image = np.array(image)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(name)
    else:
        st.error("No face found in the image!")

# Load known faces at startup
load_known_faces()

# Title and description
st.title("Face Recognition App")
st.write("Upload a photo to add to known faces or start live recognition via webcam.")

# Sidebar for adding new faces
st.sidebar.header("Add New Face")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
name = st.sidebar.text_input("Enter the person's name")

if st.sidebar.button("Add Face", key="add_face_button") and uploaded_file and name:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption=f"Uploaded image of {name}", use_column_width=True)
    add_new_face(image, name)
    save_known_faces()  # Save known faces after adding a new one
    st.sidebar.success(f"{name} has been added!")

# State to manage webcam session
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

if 'camera' not in st.session_state:
    st.session_state.camera = None

# Start webcam button
if st.button("Start Webcam for Face Recognition", key="start_webcam_button"):
    st.session_state.webcam_active = True
    st.session_state.camera = cv2.VideoCapture(0)

# Stop webcam button
if st.button("Stop Webcam", key="stop_webcam_button"):
    st.session_state.webcam_active = False
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None  # Clear camera reference
    st.empty()  # Clear the displayed frame

# Webcam live face recognition
if st.session_state.webcam_active and st.session_state.camera is not None:
    stframe = st.empty()
    camera = st.session_state.camera

    while st.session_state.webcam_active:
        success, frame = camera.read()
        if not success:
            st.error("Failed to access webcam. Please check your camera settings.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if face_distances.size > 0:
                    best_match_index = np.argmin(face_distances)
                    name = known_face_names[best_match_index] if matches[best_match_index] else "Unknown"
                else:
                    name = "Unknown"
            else:
                name = "No known faces available."

            face_names.append(name)

        # Draw rectangles around faces and display names
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Convert the frame to an image for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    # Ensure camera is released when stopping
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
