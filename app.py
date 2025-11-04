from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import cv2
import os
import numpy as np
import smtplib
from email.message import EmailMessage
from datetime import datetime

app = Flask(__name__)

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join("models", "best.pt")  # model inside repo
RESULTS_DIR = os.path.join("static", "results")  # relative path for Render storage

EMAIL_SENDER = os.getenv("EMAIL_SENDER")       # e.g. set in Render dashboard
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
# ----------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

def send_email(receiver_email, image_path, class_counts):
    """Send email with annotated image and count info"""
    msg = EmailMessage()
    msg['Subject'] = "Pantry White Wines Inventory Detection Results"
    msg['From'] = EMAIL_SENDER
    msg['To'] = receiver_email

    body = "Here are your detection results:\n\n"
    for cls, count in class_counts.items():
        body += f"{cls}: {count}\n"
    msg.set_content(body)

    # attach image
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
    msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)
    print("✅ Email sent successfully!")

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        email = request.form['email']
        file = request.files['image']

        if not file:
            return "No file uploaded!"

        img_path = os.path.join(RESULTS_DIR, file.filename)
        file.save(img_path)

        # Run inference
        results = model(img_path)
        annotated_frame = results[0].plot()
        result_path = os.path.join(RESULTS_DIR, f"annotated_{file.filename}")
        cv2.imwrite(result_path, annotated_frame)

        # Get class counts
        class_counts = {}
        names = model.names
        for box in results[0].boxes.cls.cpu().numpy():
            cls_name = names[int(box)]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                

        # Send email
        send_email(email, result_path, class_counts)

        return render_template("index.html", result_img=result_path, counts=class_counts)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
