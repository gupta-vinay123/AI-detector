import streamlit as st
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
import os

# Initialize your app
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()

# Streamlit UI
st.title("Image Classification with CNN")

st.write("Upload an image to classify:")

# File uploader to allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image to a file
    with open(clApp.filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    # Predict button
    if st.button("Classify Image"):
        result = clApp.classifier.predict()
        st.write(f"Prediction: {result}")
