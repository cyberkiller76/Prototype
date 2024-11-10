import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, Response, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Global variables
face_positions = {}
direction_count = 0
frame_id = 0
line_position = None

def track_faces(frame, previous_positions):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    new_positions = {}
    for (i, (x, y, w, h)) in enumerate(faces):
        face_center = (x + w // 2, y + h // 2)
        new_positions[i] = face_center

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return faces, new_positions

def detect_direction(prev_pos, new_pos, frame_width):
    global direction_count
    direction = None
    for face_id, prev_center in prev_pos.items():
        if face_id in new_pos:
            new_center = new_pos[face_id]
            if abs(new_center[0] - prev_center[0]) > 5:
                if new_center[0] > prev_center[0]:
                    direction = 'right'
                    if prev_center[0] < line_position <= new_center[0]:
                        direction_count += 1
                elif new_center[0] < prev_center[0]:
                    direction = 'left'
                    if prev_center[0] > line_position >= new_center[0]:
                        direction_count -= 1
    return direction

def get_crowd_level(count):
    if count <= 1:
        return 'Low'
    elif 2 <= count <= 5:
        return 'Medium'
    else:
        return 'High'

def gen_frames():
    global frame_id, face_positions, line_position, direction_count

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if line_position is None:
            line_position = frame.shape[1] // 2

        cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (0, 255, 0), 2)

        faces, new_positions = track_faces(frame, face_positions)

        if frame_id > 1:
            detect_direction(face_positions, new_positions, frame.shape[1])

        face_positions = new_positions

        crowd_level = get_crowd_level(direction_count)

        # Display the crowd level and count
        cv2.putText(frame, f"People count: {direction_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Crowd level: {crowd_level}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Encode the frame to send it to the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

def decode_base64_image(base64_str):
    # Remove the prefix "data:image/jpeg;base64,"
    img_data = base64_str.split(',')[1]
    img_data = base64.b64decode(img_data)

    # Convert binary data to image
    img = Image.open(BytesIO(img_data))
    img_np = np.array(img)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# Route to process frames sent from the client
@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        # Get the base64 image from the request
        data = request.get_json()
        base64_image = data['image']

        # Decode the image
        frame = decode_base64_image(base64_image)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode the processed frame back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"processed_image": processed_image})
    except Exception as e:
        return jsonify({"error": str(e)})




# Route for rendering the webpage
@app.route('/')
def index():
    return render_template('index.html')

# Route for streaming video frames
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, World!"})

@app.route("/api/another", methods=["GET"])
def another():
    return jsonify({"message": "This is another endpoint!"})

# Run Flask app for local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
