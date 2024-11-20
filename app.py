from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model (replace with your actual model file path)
model = tf.keras.models.load_model("Resnet50.keras")

# Define preprocessing function (adjust to your model’s input format)
def prepare_image(img):
    # Resize the image to match the model's expected input size
    img = img.resize((256, 256))  # Adjust size based on your model's requirements
    img = image.img_to_array(img)
    
    # Expand dimensions to match the input shape (batch size dimension)
    img = np.expand_dims(img, axis=0)
    
    
    img = img / 255.0  
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        
        img = Image.open(file.stream)
        img = prepare_image(img)

        prediction = model.predict(img)

        print(f"Prediction (raw output): {prediction}")

        # Get the predicted class index and map it to the label
        predicted_class_index = np.argmax(prediction, axis=1).item()
        print(f"Predicted class index: {predicted_class_index}")

        class_labels = {0: 'angry',1: 'happy',2: 'sad'}
        
        # Map the predicted index to the corresponding label
        predicted_label = class_labels.get(predicted_class_index, 'Unknown')
        print(f"Predicted label: {predicted_label}")

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
