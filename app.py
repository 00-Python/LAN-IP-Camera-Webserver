from flask import Flask, render_template, Response, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import cv2
import numpy as np


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

camera = cv2.VideoCapture(0)  # use 0 for web camera
# Load pre-trained model for person detection
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

# The User loading function required by Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized_callback():
    return redirect(url_for('login'))

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize the frame to 300x300 for SSD
            frame_resized = cv2.resize(frame, (300, 300))

            # Prepare the frame for the SSD model
            blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)

            # Pass the blob through the network
            net.setInput(blob)
            detections = net.forward()

            # Iterate over the detections
            for i in np.arange(0, detections.shape[2]):
                # Extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence > 0.2:
                    # Extract the index of the class label from the `detections`
                    idx = int(detections[0, 0, i, 1])

                    # If the class label is a person, we will draw a bounding box around it
                    if idx == 15:
                        # Compute the (x, y)-coordinates of the bounding box for the object
                        box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Draw the bounding box around the detected object on the frame
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

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


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port='5000')
