from typing import Union
from fastapi import FastAPI, Request
import os
# İgnore Warnings
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
import pickle
cwd = os.getcwd()
file_name = cwd+"/model.pkl"


app = FastAPI()


@app.get("/")
def read_root():
    
    return {"Hello": "World"}



# def cosine_similarity(teacher,student):
#     A = get_embedding(teacher, model='text-embedding-3-large')
#     B = get_embedding(student, model='text-embedding-3-large')
#     cosine = np.dot(A,B)/(norm(A)*norm(B))
#     return cosine

@app.post("/predict")
async def get_prediction(request: Request):
    # load
    model_loaded = pickle.load(open(file_name, "rb"))
    # test
    # prediction = model_loaded.predict([[0,0,0,0,0,0,0,0,0,0]])[0] 
    # print(prediction)
    message = await request.json()
    category = {"gender":["Female","Male"],"hypertension":["No","Yes"],"heart_disease":["No","Yes"],
                "ever_married":["No","Yes"],"work_type":['Never_worked','Children','Self-employed','Private','Govt_job'],"Residence_type":['Urban','Rural'],
                "smoking_status":['Unknown','Never smoked','Formerly smoked','Smokes']}
    print(category["gender"].index(message["gender"]))
    data = [[category["gender"].index(message["gender"]), message["age"], 
              category["hypertension"].index(message["hypertension"]), category["heart_disease"].index(message["heart_disease"]),
              category["ever_married"].index(message["ever_married"]), category["work_type"].index(message["work_type"]),
              category["Residence_type"].index(message["Residence_type"]), message["avg_glucose_level"], message["bmi"], category["smoking_status"].index(message["smoking_status"])]]
    prediction = model_loaded.predict(data)[0]

    if prediction == 0:
        result = {"response":"NO"}
    else:
        result = {"response":"YES"}
    return result #await request.json()
