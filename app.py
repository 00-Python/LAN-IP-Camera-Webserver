from flask import Flask, render_template, Response, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import cv2
import numpy as np
import uuid
from datetime import datetime
import base64
import requests


# Load the Haar cascade xml file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

camera = cv2.VideoCapture(0)  # use 0 for web camera

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class FaceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    face_image = db.Column(db.LargeBinary)
    body_image = db.Column(db.LargeBinary)
    eyes_image = db.Column(db.LargeBinary)
    mouth_image = db.Column(db.LargeBinary)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    unique_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()))
    gps_location = db.Column(db.String(100))  # Assuming GPS location is a string


facial_recognition_enabled = False

@app.route('/clear_db', methods=['POST'])
@login_required
def clear_db():
    # Delete all records from FaceRecord table
    try:
        num_face_records_deleted = db.session.query(FaceRecord).delete()
        db.session.commit()
        message = f"Deleted {num_face_records_deleted} face records from the database."
    except Exception as e:
        db.session.rollback()
        message = f"Error clearing image database: {e}"
    return redirect(url_for('index', message=message))


@app.route('/toggle_facial_recognition', methods=['POST'])
@login_required
def toggle_facial_recognition():
    global facial_recognition_enabled
    facial_recognition_enabled = not facial_recognition_enabled
    return redirect(url_for('index'))

# The User loading function required by Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized_callback():
    return redirect(url_for('login'))

def save_face_record(frame, bounding_box, gps_location):
    # Extract face image from the frame using the bounding box
    face_image = frame[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]

    # For body, eyes, and mouth detection, you would need to use appropriate Haar cascades or other methods
    # For example, to detect eyes within the face region:
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(face_image)

    # Assuming we take the first detected pair of eyes for simplicity
    if len(eyes) > 0:
        ex, ey, ew, eh = eyes[0]
        eyes_image = face_image[ey:ey+eh, ex:ex+ew]
    else:
        eyes_image = None

    # Similar approach for mouth detection using a mouth Haar cascade
    # And for body detection, you would need a full-body Haar cascade or another method

    # Convert images to binary data
    _, face_image_encoded = cv2.imencode('.png', face_image)
    eyes_image_encoded = cv2.imencode('.png', eyes_image)[1] if eyes_image is not None else None
    # Repeat for mouth and body images

    # Create a new FaceRecord object
    face_record = FaceRecord(
        face_image=face_image_encoded.tobytes(),
        eyes_image=eyes_image_encoded.tobytes() if eyes_image_encoded is not None else None,
        # Repeat for mouth and body images
        gps_location=gps_location
    )

    # Add the new face record to the database
    db.session.add(face_record)
    db.session.commit()

def get_gps_location(api_key="6bc9fc19816243999549491eae7c3aef"):
    response = requests.get(f'https://api.ipgeolocation.io/ipgeo?apiKey={api_key}')
    if response.status_code == 200:
        data = response.json()
        return f"{data['latitude']},{data['longitude']}"
    else:
        return "0.0,0.0"  # Default value in case of an error

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if facial_recognition_enabled:
                # Convert frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

                # Iterate over the face detections
                for (x, y, w, h) in faces:
                    bounding_box = (x, y, w, h)

                    # Save the face record to the database
                    with app.app_context():
                        # Define a dummy GPS location for demonstration purposes
                        dummy_gps_location = "0.0,0.0"
                        save_face_record(frame, bounding_box, dummy_gps_location)

                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Encode the frame regardless of facial recognition state
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/video')
@login_required
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))

        return render_template('login.html', error_message='Invalid username or password')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the username already exists in the database
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error_message='Username already exists')

        # Create a new User object and set the username and password
        new_user = User(username=username)
        new_user.set_password(password)

        # Add the new user to the database
        db.session.add(new_user)
        db.session.commit()

        # Redirect to the login page
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/image_database')
@login_required
def image_database():
    # Query all FaceRecord entries from the database
    face_records = FaceRecord.query.all()
    # Convert binary image data to base64 for HTML display
    for record in face_records:
        record.face_image_base64 = base64.b64encode(record.face_image).decode('ascii')
    # Render the template with the face records
    return render_template('image_database.html', face_records=face_records)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port='5000')
