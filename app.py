from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Tải các mô hình đã huấn luyện
model_species = tf.keras.models.load_model('species_model.h5')
model_emotion = tf.keras.models.load_model('emotion_model.h5')  # Sửa lại tên để dùng đúng mô hình cảm xúc

# Danh sách nhãn (cập nhật theo nhãn trong tập dữ liệu)
species_labels = ['Cat', 'Dog']  # Cập nhật danh sách nhãn loài
emotion_labels = ['Angry', 'Disgusted', 'Happy', 'Relaxed', 'Sad', 'Scared', 'Surprised']  # Cập nhật danh sách nhãn cảm xúc

# Hàm dự đoán loài vật và cảm xúc
def predict_species_and_emotion(image_path):
    img = Image.open(image_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán loài vật
    species_pred = model_species.predict(img_array)
    species_class = np.argmax(species_pred)
    species_result = species_labels[species_class]

    # Dự đoán cảm xúc
    emotion_pred = model_emotion.predict(img_array)
    emotion_class = np.argmax(emotion_pred)
    emotion_result = emotion_labels[emotion_class]

    return species_result, emotion_result

# Route cho trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            species, emotion = predict_species_and_emotion(filepath)
            return render_template('index.html', species=species, emotion=emotion, image_path=filepath)
    return render_template('index.html')

# Route phục vụ file tĩnh
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
