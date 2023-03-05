# %%
# LOAD MODEL
from joblib import load

filename = "myFirstSavedModel.joblib"

clfUploaded = load(filename)
# %%
from sklearn.datasets import load_iris


dataSet = load_iris()

labelsNames = list(dataSet.target_names)
# %%

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import numpy as np

templates = Jinja2Templates(directory="templates")

app = FastAPI()


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html",
                                      {"request": request})


@app.get("/predict/")
async def make_prediction(request: Request, l1: float,
                          w1: float,
                          l2: float, w2: float):
    testData = np.array([l1, w1, l2, w2]).reshape(-1, 4)
    probalities = clfUploaded.predict_proba(testData)[0]
    predicted = np.argmax(probalities)
    probabilty = probalities[predicted]
    predicted = labelsNames[predicted]

    return templates.TemplateResponse("prediction.html",
                                      {"request": request,
                                       "probalities": probalities,
                                       "predicted": predicted,
                                       "probabilty": probabilty})
