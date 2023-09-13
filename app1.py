from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = "C:\\Users\\jbram\\Desktop\\Project\\Face rr\\UPLOAD_FOLDER"

# Data structure to store registered faces and associated data
registered_faces = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # type: ignore

@app.route("/")
def home(): # type: ignore
    return render_template("index1.html")

@app.route("/register", methods=["POST"])
def register_face():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    try:
        # Capture additional information (e.g., name) from the user
        name = request.form.get("name")
        if not name:
            return jsonify({"error": "Name is required"})

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], name + ".jpg")
        file.save(img_path)

        # Store the registered face and associated data
        registered_faces.append({"name": name, "image_path": img_path})

        return jsonify({"message": f"Face for {name} registered successfully"})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        recognition_method = request.form.get("recognition_method")

        if recognition_method == "webcam":
            # Initialize the webcam
            cap = cv2.VideoCapture(0)

            while True:
                # Capture frame from the webcam
                ret, frame = cap.read()

                # Save the webcam frame as a temporary image
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg")
                cv2.imwrite(img_path, frame)

                result = {"message": "Face not recognized"}

                for registered_face in registered_faces:
                    try:
                        obj = DeepFace.verify(img_path, registered_face["image_path"], enforce_detection=False)
                        if obj["verified"]:
                           result = {"message": f"Face recognized as {registered_face['name']} with {obj['distance']} distance"}
                           break  # Break the loop when a match is found
                    except Exception as e:
                        continue

                # Display the recognition result on the webcam feed
                cv2.putText(frame, result["message"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Recognition', frame)
                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the webcam and close OpenCV windows
            cap.release()
            cv2.destroyAllWindows()

        elif recognition_method == "upload":
            if "file" not in request.files:
                return jsonify({"error": "No file part"})

            file = request.files["file"]

            if file.filename == "":
                return jsonify({"error": "No selected file"})

            # Read and process the uploaded image
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert the image to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save or display the processed image with detected faces
            cv2.imwrite("static/processed_image.jpg", img)

        else:
            return jsonify({"error": "Invalid recognition method"})

        return jsonify({"message": "Recognition completed"})

    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
    app.run(debug=True)
