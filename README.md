# Traffic Sign Recognition Using CNN

A Python-based traffic sign classification system using a Convolutional Neural Network (CNN) and the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The system classifies traffic sign images and displays the predicted sign through a graphical user interface (GUI), with speech output for the recognized sign.

## Technologies Used

- Python
- TensorFlow / Keras
- CNN
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Tkinter

## Project Structure

```text
Traffic-Sign-Recognition/
├── datasettraffic/
├── traffic_model.ipynb
├── my_model.h5
├── signs.py
├── Front-end.py
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/MahekShikalgar2/Traffic-Sign-Recognition.git
cd Traffic-Sign-Recognition
```

Install the required libraries:

```bash
pip install opencv-python numpy tensorflow keras matplotlib scikit-learn pillow
```

## Usage

1. Open `traffic_model.ipynb` and run the notebook to train the model.
2. Ensure the trained model is saved as `my_model.h5`.
3. Run the GUI application:

```bash
python Front-end.py
```

4. Select a traffic sign image to view its predicted class and hear the sign name.

## Objective

To recognize traffic signs using deep learning and provide predictions through a simple graphical interface with audio feedback.

---

