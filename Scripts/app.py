# Import necessary modules
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import subprocess
import os

# Create the Flask app
app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'Project_Yolov8/Datasets/documentOcr/test/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Counter for uploaded images
uploaded_images_count = 0

# Route for the home page
@app.route('/')
def index():
    return render_template('PDF-TXT.html', uploaded_images_count=uploaded_images_count)

# Route for uploading a file
@app.route('/upload_file', methods=['POST'])
def upload_file():
    global uploaded_images_count

    try:
        if 'file' not in request.files:
            return jsonify({'upload_result': 'No file'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'upload_result': 'No selected file'})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        uploaded_images_count += 1

        return jsonify({'upload_result': 'File uploaded successfully'})
    except Exception as e:
        return jsonify({'upload_result': str(e)})

# Route for predicting with YOLOv8
@app.route('/predict_yolov8')
def predict_yolov8():
    result = predict_with_yolov8()
    return render_template('PDF-TXT.html', yolov8_result=result, uploaded_images_count=uploaded_images_count)

# Function to call YOLOv8 script
def predict_with_yolov8():
    try:
        # เปลี่ยนไดเรกทอรีทำงานปัจจุบัน
        os.chdir(os.path.join(os.getcwd(), 'Project_Yolov8'))

        # คำสั่งเรียกสคริปต์ YOLOv8
        command = [
            'python', 'yolov8_documentOcr_SAHI_old.py', 'predict', 'yolov8_documentOcr_thai-dataset.yaml',
            'TrainResult/documentOcr/weights/best.pt', 'Datasets/documentOcr/test/images', 'auto', 'save'
        ]

        
     
        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, cwd=os.getcwd())

        # Check for errors
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        # Return the result
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
