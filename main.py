from typing import Union
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
# İgnore Warnings
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
import pickle
cwd = os.getcwd()
file_name = cwd+"/model.pkl"


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    
    return {"Hello": "World"}


@app.post("/predict")
async def get_prediction(request: Request):
    # load
    model_loaded = pickle.load(open(file_name, "rb"))
    # test
    # prediction = model_loaded.predict([[0,0,0,0,0,0,0,0,0,0]])[0] 
    # print(prediction)
    message = await request.json()
    print(message)
    category = {"gender":["Female","Male"],"hypertension":["No","Yes"],"heart_disease":["No","Yes"],
                "ever_married":["No","Yes"],"work_type":['Never_worked','Children','Self-employed','Private','Govt_job'],"Residence_type":['Urban','Rural'],
                "smoking_status":['Unknown','Never smoked','Formerly smoked','Smokes']}
    # print(category["gender"].index(message["gender"]))
    data = [[category["gender"].index(message["gender"]), int(message["age"]), 
              category["hypertension"].index(message["hypertension"]), category["heart_disease"].index(message["heart_disease"]),
              category["ever_married"].index(message["ever_married"]), category["work_type"].index(message["work_type"]),
              category["Residence_type"].index(message["Residence_type"]), int(message["avg_glucose_level"]), float(message["bmi"]), category["smoking_status"].index(message["smoking_status"])]]
    prediction = model_loaded.predict(data)[0]
    print(prediction)
    
    if (message["hypertension"]=='Yes' or message["heart_disease"]=='Yes') and (message["smoking_status"]=='Smokes' or message["smoking_status"]=='Formerly smoked') and int(message["avg_glucose_level"])>= 125 and float(message["bmi"])>= 25:
        result = {"response":"YES"}
    elif message["hypertension"]=='Yes' and message["heart_disease"]=='Yes' and (message["smoking_status"]=='Smokes' or message["smoking_status"]=='Formerly smoked'):
        result = {"response":"YES"}
    elif message["hypertension"]=='Yes' and message["heart_disease"]=='Yes':
        result = {"response":"YES"}
    elif prediction == 0:
        result = {"response":"NO"}
    else:
        result = {"response":"NO"}
    return result 
