# CNN-Rice_Grain_classifier

#  Rice Grain Classifier

A machine learning web app that classifies rice grain images into five types using a deep learning model. Built with TensorFlow and deployed using Streamlit.


##  Project Structure

rice-grain-classifier/
│
├── app.py                  # Streamlit frontend
├── main.py                 # Optional core logic/utility functions
├── train.py                # Model training script
├── evaluate.py             # Model evaluation script
├── models/
│   └── rice_classifier.h5  # Trained model
├── Untitled.png            # Project/logo image
└── README.md               # You're reading it!



##  Setup Instructions

### 1. Clone the repository

bash
git clone https://github.com/your-username/rice-grain-classifier.git
cd rice-grain-classifier


### 2. Create a virtual environment (optional but recommended)

bash
python -m venv venv
on Windows: venv\Scripts\activate


### 3. Install dependencies

bash
pip install -r requirements.txt


If you don’t have a `requirements.txt` file yet, you can create one like this:

bash
pip install streamlit tensorflow pillow numpy
pip freeze > requirements.txt


## Training the Model
To train the model from scratch using your dataset:

bash
python train.py


## Evaluating the Model
To evaluate the trained model and check its accuracy:

bash
python evaluate.py

Ensure evaluate.py loads models/rice_classifier.h5 and the test data is in place.


##  Running the Application

bash
streamlit run app.py


##  Usage

1. Launch the app.
2. Upload an image of a rice grain.
3. The model will predict and display the type: **Arborio**, **Basmati**, **Ipsala**, **Jasmine**, or **Karacadag**.


##  Model Info

- Model Type: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input Shape: (224, 224, 3)
- Output: 5-class Softmax classifier


##  Acknowledgements

- Rice Image Dataset used for model training.
- Streamlit and TensorFlow communities for their amazing tools.
