# Facial Emotion Recognition

This project demonstrates **facial emotion recognition using a Convolutional Neural Network (CNN)** in Python. The system is trained on the **FER-2013 dataset** and can recognize human emotions from facial images. It also supports **real-time emotion recognition using a webcam**.

## Features

* Facial emotion classification using a CNN
* Training on the FER-2013 dataset
* Real-time facial emotion recognition using a webcam
* OpenCV-based face detection
* Model saving and loading using Keras/TensorFlow

## Emotions

The model is trained to recognize the following seven emotions:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## Prerequisites

Make sure you have **Python 3.x** installed.

The following Python libraries are required:

* NumPy
* Pandas
* OpenCV
* Keras
* TensorFlow
* Scikit-learn

Install all dependencies using:

```bash
pip install numpy pandas opencv-python keras tensorflow scikit-learn
```

## Dataset

This project uses the **FER-2013 (Facial Expression Recognition 2013)** dataset.

You can download the dataset from Kaggle:

**FER-2013 Dataset:**
https://www.kaggle.com/datasets/msambare/fer2013

After downloading the dataset, place the `fer2013.csv` file in the project directory.

The project structure should look similar to:

```text
Facial-Emotion-Recognition/
│
├── fer2013.csv
├── train_emotion_model.py
├── emotion_recognition.py
└── README.md
```

## Usage

### 1. Train the Model

Run the following command to train the CNN model:

```bash
python train_emotion_model.py
```

After training is completed, the trained model will be saved as:

```text
emotion_model.h5
```

### 2. Run Real-Time Emotion Recognition

Start the webcam-based emotion recognition system using:

```bash
python emotion_recognition.py
```

The program will access your webcam, detect faces, and predict the emotion displayed on each detected face.

Press:

```text
q
```

to exit the webcam window.

## Model Architecture

The project uses a **Convolutional Neural Network (CNN)** for emotion classification.

The architecture consists of:

* Convolutional layers for extracting facial features
* Max-pooling layers for dimensionality reduction
* Dropout layers to reduce overfitting
* Fully connected (Dense) layers for classification
* Softmax output layer for predicting the emotion classes

The model is compiled using:

* **Loss:** Categorical Cross-Entropy
* **Optimizer:** Adam
* **Output:** Seven emotion classes

## Technologies Used

| Technology   | Purpose                              |
| ------------ | ------------------------------------ |
| Python       | Programming language                 |
| TensorFlow   | Deep learning framework              |
| Keras        | CNN model development                |
| OpenCV       | Face detection and webcam processing |
| NumPy        | Numerical operations                 |
| Pandas       | Dataset processing                   |
| Scikit-learn | Data preprocessing and evaluation    |
| FER-2013     | Facial emotion dataset               |

## Project Workflow

```text
FER-2013 Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
CNN Model Training
       │
       ▼
Emotion Model
(emotion_model.h5)
       │
       ▼
Webcam Input
       │
       ▼
Face Detection
       │
       ▼
Emotion Prediction
       │
       ▼
Detected Emotion
```

## Contributing

Contributions are welcome!

If you have suggestions, improvements, or bug fixes, feel free to:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Open a Pull Request

## License

This project is intended for educational and research purposes.
