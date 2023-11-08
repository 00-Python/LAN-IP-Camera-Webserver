from flask import Flask, render_template, Response, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import cv2
import numpy as np
from mtcnn import MTCNN
import uuid
from datetime import datetime
import base64


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
# Initialize the MTCNN model
detector = MTCNN()

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
    left_eye_image = db.Column(db.LargeBinary)
    right_eye_image = db.Column(db.LargeBinary)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    unique_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()))


facial_recognition_enabled = False

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

def save_face_record(frame, bounding_box, keypoints):
    # Extract face and eye images from the frame using the bounding box and keypoints
    face_image = frame[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
    left_eye_image = frame[keypoints['left_eye'][1]-10:keypoints['left_eye'][1]+10, keypoints['left_eye'][0]-10:keypoints['left_eye'][0]+10]
    right_eye_image = frame[keypoints['right_eye'][1]-10:keypoints['right_eye'][1]+10, keypoints['right_eye'][0]-10:keypoints['right_eye'][0]+10]

    # Convert images to binary data
    _, face_image_encoded = cv2.imencode('.png', face_image)
    _, left_eye_image_encoded = cv2.imencode('.png', left_eye_image)
    _, right_eye_image_encoded = cv2.imencode('.png', right_eye_image)

    # Create a new FaceRecord object
    face_record = FaceRecord(
        face_image=face_image_encoded.tobytes(),
        left_eye_image=left_eye_image_encoded.tobytes(),
        right_eye_image=right_eye_image_encoded.tobytes()
    )

    # Add the new face record to the database
    db.session.add(face_record)
    db.session.commit()


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if facial_recognition_enabled:
                # Detect faces in the frame
                result = detector.detect_faces(frame)

                # Iterate over the face detections
                for person in result:
                    bounding_box = person['box']
                    keypoints = person['keypoints']

                    # Save the face record to the database without the landmarks
                    with app.app_context():
                        save_face_record(frame, bounding_box, keypoints)

                    # Draw a rectangle around the face
                    cv2.rectangle(frame,
                                (bounding_box[0], bounding_box[1]),
                                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                                (0,155,255),
                                2)

                    # Draw the facial landmarks
                    cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
                    cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
                    cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
                    cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
                    cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)

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
# @login_required
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
        record.left_eye_image_base64 = base64.b64encode(record.left_eye_image).decode('ascii')
        record.right_eye_image_base64 = base64.b64encode(record.right_eye_image).decode('ascii')
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
