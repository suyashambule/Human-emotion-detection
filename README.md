## Human Emotion Detection

This project involves the development of a machine learning model that can classify human emotions from text, speech, or images. The goal is to predict the emotional state of a person based on various input data such as text (e.g., tweets, reviews), speech (e.g., voice recordings), or images (e.g., facial expressions).

![alt text](Human-emotion-detection/images/Screenshot 2024-11-20 at 6.35.19 PM.png)


Features

Emotion detection from text, speech, and image data
Multi-class classification (e.g., happy, sad, angry)
Trained using state-of-the-art machine learning models
Real-time emotion prediction
Easy-to-use API or interface for integration into applications
Technologies Used

Python: Programming language used for model development.
TensorFlow / PyTorch: Deep learning frameworks used for training and deploying the emotion detection models.
Keras: High-level neural networks API used for building deep learning models.
Scikit-learn: For data preprocessing, model evaluation, and classification tasks.
OpenCV / Dlib: For image processing and detecting emotions from facial expressions.
SpeechRecognition / librosa: For emotion detection from speech.
Flask / FastAPI: Web frameworks for deploying the model as a REST API.
Installation

Follow these steps to get the project up and running on your local machine.

Clone the repository:
git clone https://github.com/suyashambule/emotion-detection.git
cd emotion-detection
Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:
pip install -r requirements.txt
The project includes a REST API for emotion detection. You can run the API using:

python app.py
The API will be available at http://localhost:5000. You can send POST requests with text, image, or audio data to get emotion predictions.

Model

The model is trained on [Human-emotion-dataset] avaliable on kaggle and can classify emotions like:

Happy
Sad
Angry

The model uses [VGG19 and many more], which is optimized for high accuracy and performance.

Evaluation
![alt text](Human-emotion-detection/images/Screenshot 2024-11-20 at 6.35.19 PM.png)



Contributions are welcome! If you would like to improve this project, feel free to fork the repository, create a branch, and submit a pull request.



Acknowledgements

The AffectNet Dataset for facial emotion recognition.
The RAVDESS Dataset for speech emotion recognition.
OpenCV for facial expression detection.
Images

Description of the first image, such as the flow of emotion detection from text, image, or audio.
Description of the second image, such as the architecture of the machine learning model used for emotion detection.
Description of the third image, such as a sample output of emotion predictions from the system.