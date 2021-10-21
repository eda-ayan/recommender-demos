import json
import numpy as np
import os
import joblib

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "./movie.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    user_id = int(np.array(json.loads(raw_data)["user_id"]))
    movie_id = int(np.array(json.loads(raw_data)["movie_id"]))
    # make prediction
    predicted = model.predict(user_id, movie_id)
    # you can return any data type as long as it is JSON-serializable
    return predicted