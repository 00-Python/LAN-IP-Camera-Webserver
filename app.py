from flask import Flask, render_template, Response, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Session
from flask_migrate import Migrate

import cv2
import numpy as np
import uuid
from datetime import datetime
import base64
import requests
from scipy.stats import pearsonr
from flask import jsonify


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)

camera = cv2.VideoCapture(0)

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
    profile_id = db.Column(db.Integer, db.ForeignKey('profile.id'))
    body_image = db.Column(db.LargeBinary)
    face_image = db.Column(db.LargeBinary)
    left_eye_image = db.Column(db.LargeBinary)
    right_eye_image = db.Column(db.LargeBinary)
    mouth_image = db.Column(db.LargeBinary)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    unique_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()))
    gps_location = db.Column(db.String(100))
    profile = db.relationship('Profile', back_populates='face_records')  # Modify this line

class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    face_records = db.relationship('FaceRecord', back_populates='profile', lazy=True)  # Modify this line


facial_recognition_enabled = False

def create_profiles(correlation_threshold: float = 0.95):
    all_face_records = FaceRecord.query.all()
    profiles = []

    for record in all_face_records:
        if not record.profile_id:
            # Create a new profile for this record
            profile = Profile(name=f"Profile {len(profiles) + 1}")
            db.session.add(profile)
            record.profile = profile
            profiles.append(profile)
            db.session.commit()  # Commit after adding each profile

            for other_record in all_face_records:
                if record.id != other_record.id and not other_record.profile_id:
                    image1 = cv2.imdecode(np.frombuffer(record.face_image, np.uint8), cv2.IMREAD_COLOR)
                    image2 = cv2.imdecode(np.frombuffer(other_record.face_image, np.uint8), cv2.IMREAD_COLOR)
                    classification = classify_by_pearson(image1, image2, correlation_threshold)
                    if classification == 'Similar':
                        other_record.profile = record.profile
                        db.session.commit()  # Commit after classifying each record


def classify_by_pearson(image1, image2, correlation_threshold: int):
    # Resize images to the same size if they are different
    if image1.shape != image2.shape:
        # Choose a common size, for example, the size of the first image
        target_size = image1.shape[:2]
        image2 = cv2.resize(image2, (target_size[1], target_size[0]))

    # Flatten the images to 1D arrays
    image1_flattened = image1.flatten()
    image2_flattened = image2.flatten()

    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(image1_flattened, image2_flattened)

    # Classify based on the correlation
    if correlation > correlation_threshold:  # This threshold can be adjusted
        return 'Similar'
    else:
        return 'Different'

def save_face_record(frame, bounding_box, gps_location):
    # Save the whole or partial body image
    body_image = frame[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
    _, body_image_encoded = cv2.imencode('.png', body_image)

    # Detect the face within the body image
    gray_body = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_body, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        face_image = body_image[fy:fy+fh, fx:fx+fw]
        _, face_image_encoded = cv2.imencode('.png', face_image)

        # Detect eyes within the face image
        eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eyes_cascade.detectMultiScale(face_image)

        left_eye_image = None
        right_eye_image = None
        eyes = sorted(eyes, key=lambda x: x[0])

        if len(eyes) >= 2:
            left_eye = eyes[0]
            right_eye = eyes[1]

            lex, ley, lew, leh = left_eye
            left_eye_image = face_image[ley:ley+leh, lex:lex+lew]
            _, left_eye_image_encoded = cv2.imencode('.png', left_eye_image)

            rex, rey, rew, reh = right_eye
            right_eye_image = face_image[rey:rey+reh, rex:rex+rew]
            _, right_eye_image_encoded = cv2.imencode('.png', right_eye_image)

        # Detect mouth within the face image
        mouth_image = None
        mouth_rects = mouth_cascade.detectMultiScale(gray_body[fy:fy+fh, fx:fx+fw], scaleFactor=1.5, minNeighbors=20, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(mouth_rects) > 0:
            mx, my, mw, mh = mouth_rects[0]
            mouth_image = face_image[my:my+mh, mx:mx+mw]
            _, mouth_image_encoded = cv2.imencode('.png', mouth_image)

        # Only proceed if all images are present
        if face_image is not None and left_eye_image is not None and right_eye_image is not None and mouth_image is not None:
            face_record = FaceRecord(
                body_image=body_image_encoded.tobytes(),
                face_image=face_image_encoded.tobytes(),
                left_eye_image=left_eye_image_encoded.tobytes(),
                right_eye_image=right_eye_image_encoded.tobytes(),
                mouth_image=mouth_image_encoded.tobytes(),
                gps_location=gps_location
            )

            db.session.add(face_record)
            db.session.commit()


def get_gps_location(api_key="6bc9fc19816243999549491eae7c3aef"):
    response = requests.get(f'https://api.ipgeolocation.io/ipgeo?apiKey={api_key}')
    if response.status_code == 200:
        data = response.json()
        return f"{data['latitude']},{data['longitude']}"
    else:
        return "0.0,0.0"

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if facial_recognition_enabled:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

                for (x, y, w, h) in faces:
                    bounding_box = (x, y, w, h)

                    with app.app_context():
                        dummy_gps_location = "0.0,0.0"
                        save_face_record(frame, bounding_box, dummy_gps_location)  # Save the whole or partial body image


                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    mouth_rects = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=20, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

                    for (mx, my, mw, mh) in mouth_rects:
                        cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@login_manager.user_loader
def load_user(user_id):
    with Session(db.engine) as session:
        return session.get(User, int(user_id))

@login_manager.unauthorized_handler
def unauthorized_callback():
    return redirect(url_for('login'))


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

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error_message='Username already exists')

        new_user = User(username=username)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/toggle_facial_recognition', methods=['POST'])
@login_required
def toggle_facial_recognition():
    global facial_recognition_enabled
    facial_recognition_enabled = not facial_recognition_enabled
    return redirect(url_for('index'))

@app.route('/image_database')
@login_required
def image_database():
    face_records = FaceRecord.query.all()
    for record in face_records:
        record.face_image_base64 = base64.b64encode(record.face_image).decode('ascii') if record.face_image else None
        record.left_eye_image_base64 = base64.b64encode(record.left_eye_image).decode('ascii') if record.left_eye_image else None
        record.right_eye_image_base64 = base64.b64encode(record.right_eye_image).decode('ascii') if record.right_eye_image else None
        record.mouth_image_base64 = base64.b64encode(record.mouth_image).decode('ascii') if record.mouth_image else None
    return render_template('image_database.html', face_records=face_records)

@app.route('/clear_face_records', methods=['POST'])
@login_required
def clear_face_records():
    try:
        num_face_records_deleted = db.session.query(FaceRecord).delete()
        db.session.commit()
        message = f"Deleted {num_face_records_deleted} face records from the database."
    except Exception as e:
        db.session.rollback()
        message = f"Error clearing face records database: {e}"
    return redirect(url_for('image_database', message=message))

@app.route('/clear_profiles', methods=['POST'])
@login_required
def clear_profiles():
    try:
        num_profiles_deleted = db.session.query(Profile).delete()
        db.session.commit()
        message = f"Deleted {num_profiles_deleted} profiles from the database."
    except Exception as e:
        db.session.rollback()
        message = f"Error clearing profiles database: {e}"
    return redirect(url_for('classify', message=message))


@app.route('/compare_faces', methods=['POST'])
@login_required
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return 'Missing images', 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        return 'No selected file', 400

    try:
        image1 = cv2.imdecode(np.fromstring(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.fromstring(file2.read(), np.uint8), cv2.IMREAD_COLOR)

        classification = classify_by_pearson(image1, image2)

        return jsonify({'classification': classification})
    except Exception as e:
        return str(e), 500

@app.route('/classify', methods=['GET', 'POST'])
@login_required
def classify():
    if request.method == 'POST':
        correlation_threshold = float(request.form.get('correlation_threshold', 0.85))
        create_profiles(correlation_threshold)  # Pass the threshold to the function
    else:
        correlation_threshold = 0.85  # Default value
    all_profiles = Profile.query.all()
    for profile in all_profiles:
        for record in profile.face_records:
            record.face_image_base64 = base64.b64encode(record.face_image).decode('ascii') if record.face_image else None
    return render_template('classify.html', profiles=all_profiles, correlation_threshold=correlation_threshold)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port='5000')
