import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil


class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename

    def model_copy(self):
        source_path = "artifacts/training/model.h5"
        destination_path = "model/model.h5"
        if not os.path.exists(source_path):
             return f" Model file {source_path} does not exist."
        
        destination_dir = os.path.dirname(destination_path)

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir,exist_ok=True)
        if not os.path.exists(destination_path):
            shutil.copy(source_path, destination_path)
    
    def predict(self):
        ## load model
        
        #model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{ "image" : prediction}]