import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import threading
import pyttsx3
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
from PIL import Image, ImageTk
import queue
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration & Initialization
emotion_counts = {"Anger": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Sad": 0, "Surprise": 0, "Neutral": 0}
emotion_trend = deque(maxlen=10)
labels = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
speech_queue = queue.Queue()

# Load Model (Ensure these files are in the same folder)
try:
    model = load_model('facialemotionmodel.h5')
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
except Exception as e:
    print(f"Error loading model/cascade: {e}")

# Speech Engine Thread
def provide_voice_feedback():
    engine = pyttsx3.init()
    while True:
        emotion = speech_queue.get()
        if emotion == 'QUIT': break
        engine.say(f"You are feeling {emotion}")
        engine.runAndWait()

threading.Thread(target=provide_voice_feedback, daemon=True).start()

# GUI Setup
root = tk.Tk()
root.title("BCA Project: Facial Emotion Recognition")
root.geometry("1000x900")
root.config(bg="#2e3b4e")

label_prediction = tk.Label(root, text="Real-time Emotion Detection", font=('Helvetica', 18, 'bold'), fg="white", bg="#2e3b4e")
label_prediction.pack(pady=10)

label_frame = tk.Label(root)
label_frame.pack(padx=20, pady=10)

label_emotion_count = tk.Label(root, text="Most Detected: None", font=('Helvetica', 14), fg="white", bg="#2e3b4e")
label_emotion_count.pack(pady=5)

webcam = cv2.VideoCapture(0)

def update_frame():
    ret, frame = webcam.read()
    if not ret: return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detected_this_frame = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
        roi_gray = roi_gray.reshape(1, 48, 48, 1)
        
        preds = model.predict(roi_gray)
        emotion = labels[np.argmax(preds)]
        detected_this_frame.append(emotion)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for e in detected_this_frame:
        emotion_counts[e] += 1
        if speech_queue.empty(): # Avoid backlog
            speech_queue.put(e)

    if detected_this_frame:
        most_detected = max(emotion_counts, key=emotion_counts.get)
        label_emotion_count.config(text=f"Most Detected Emotion: {most_detected}")
        update_bg(most_detected)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(image=img)
    label_frame.img_tk = img_tk
    label_frame.configure(image=img_tk)
    root.after(10, update_frame)

def update_bg(emotion):
    colors = {"Happy": "#28a745", "Sad": "#007bff", "Anger": "#dc3545", "Neutral": "#6c757d"}
    root.config(bg=colors.get(emotion, "#2e3b4e"))

# Buttons
tk.Button(root, text="Show History", command=lambda: messagebox.showinfo("History", str(emotion_counts))).pack(pady=5)
tk.Button(root, text="Quit", command=root.destroy, bg="#cc0000", fg="white").pack(pady=5)

update_frame()
root.mainloop()
webcam.release()
speech_queue.put('QUIT')
