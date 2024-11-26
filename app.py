from flask import Flask, request, render_template
from src.predict import process_large_file
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    anomaly_count, type_distribution = process_large_file(file_path)
    return render_template(
        'result.html',
        anomaly_count=anomaly_count,
        type_distribution=type_distribution
    )

if __name__ == '__main__':
    app.run(debug=True)
