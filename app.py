from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logger as LOG
from src.utils import load_object
from src.exceptions import CustomException
import sys

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
   return templates.TemplateResponse(
    request=request,
    name="index.html"
)


@app.post("/predict")
async def predict(
    gender: str = Form(...),
    age: int = Form(...),
    hypertension: int = Form(0),
    heart_disease: int = Form(0),
    smoking_history: str = Form(...),
    bmi: float = Form(...),
    HbA1c_level: float = Form(...),
    blood_glucose_level: int = Form(...)
):
    try:
        # Create CustomData object with form inputs
        custom_data = CustomData(
            gender=gender,
            age=age,
            hypertension=hypertension,
            heart_disease=heart_disease,
            smoking_history=smoking_history,
            bmi=bmi,
            HbA1c_level=HbA1c_level,
            blood_glucose_level=blood_glucose_level
        )
        
        features = custom_data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(features)
        
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        LOG.info(f"Prediction result: {result}")
        
        return JSONResponse({"prediction": result})
    
    except Exception as e:
        LOG.error(f"Error during prediction: {str(e)}")
        return JSONResponse({"error": f"Error during prediction: {str(e)}"}, status_code=500)