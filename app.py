import joblib as jb
import pandas as pd
from fastapi import FastAPI
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

with open('model.pkl', 'rb') as f:
    model = jb.load(f)

app = FastAPI()

@app.get('/')
def home():
    return {'message': 'Hello'}


class UserInput(BaseModel):
    time_spent_alone: Annotated[float, Field(..., gt=0, description='time spent alone')]
    stage_fear: Annotated[Literal['Yes', 'No'], Field(..., description='You have stage fear or not')]
    social_event_attendance: Annotated[float, Field(..., gt=0, description='Social event attendance')]
    going_outside: Annotated[float, Field(..., gt=0, description='Going outside')]
    drained_after_socializing: Annotated[Literal['Yes', 'No'], Field(..., description='Have you ever drained after socializing')]
    friends_circle_size: Annotated[float, Field(..., gt=0, description='Friend circle size')]
    post_frequency: Annotated[float, Field(..., gt=0, description='Post Frequency')]


@app.post('/predict')
def prediction(user_data: UserInput):

    input = pd.DataFrame([{
        'time_spent_alone': user_data.time_spent_alone,
        'stage_fear': user_data.stage_fear,
        'social_event_attendance': user_data.social_event_attendance,
        'going_outside': user_data.going_outside,
        'drained_after_socializing': user_data.drained_after_socializing,
        'friends_circle_size': user_data.friends_circle_size,
        'post_frequency': user_data.post_frequency
    }])

    prediction = model.predict(input)
    raise JSONResponse(200, content= {'prediction': prediction})