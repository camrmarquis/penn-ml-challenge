from fastapi import FastAPI, HTTPException
from fastapi.logger import logger
from pydantic import BaseModel
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from typing import Dict
import uvicorn
from pickle import load
import pandas as pd


# Set the log level to what is configured in the environment. Default to 'DEBUG'
logger.setLevel(os.getenv('LOG_LEVEL', 'DEBUG'))

# create app instance
app = FastAPI()

model = load_model("model.h5", compile=False)
scaler = load(open('scaler.pkl', 'rb'))

# Build request schemas


class PredictionDataIn(BaseModel):
    bitcoin_last_minute: Dict

# Helper functions


def sort_prediction_request(feature_dict):
    """
    Ensures the keys are in correct order per observation

    :param feature_dict: input dictionary from incoming request
    :return: sorted dict
    """
    i = 0

    new = dict()
    while i < len(feature_dict.values()) - 1:

        if i == 0:
            sort_values = ['price_open', 'price_low', 'price_close', 'volume_traded', 'trades_count']
            keys = list(feature_dict.keys())[i:i + 5]
            values = list(feature_dict.values())[i:i + 5]
            temp = dict(zip(keys, values))
            tail = ''
            value_order = [temp[key] for key in sort_values]
            keys = [key + tail for key in sort_values]
            sort = dict(zip(keys, value_order))

            new.update(sort)
            i += 5
        else:
            sort_values = ['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded', 'trades_count']
            keys = ['_'.join(key.split('_')[:-1]) for key in list(feature_dict.keys())[i:i + 6]]
            values = list(feature_dict.values())[i:i + 6]
            temp = dict(zip(keys, values))
            tail = list(feature_dict.keys())[i].split('_')[-1]
            print(tail)
            value_order = [temp[key] for key in sort_values]
            keys = [key + '_' + tail for key in sort_values]
            sort = dict(zip(keys, value_order))

            new.update(sort)
            i += 6

    return new


def build_series_df(sorted_data):
    """
    Create DatFrame containing the time series structure for prediction

    :param sorted_data: Sorted data from input request
    :return: Pandas dataframe
    """
    i = 0
    j = 0
    df = pd.DataFrame(index=range(0, 59), columns=['price_open', 'price_high', 'price_low',
                                                   'price_close', 'volume_traded', 'trades_count'])

    while i < len(sorted_data.values()) - 1:
        if i == 0:
            i += 5
        else:
            values = list(sorted_data.values())[i:i + 6]
            df.loc[j] = values
            i += 6
            j += 1

    return df


@app.get("/", status_code=200)
async def root():
    return {"message": "Hello Penn"}


@app.get("/health", status_code=200)
async def health_check():
    # Usually there would be status on databases, caches, etc here but you get the jist
    return {"status": "happy healthy app"}


@app.post("/predict", status_code=200)
async def predict_bitcoin_next_min(request: PredictionDataIn):

    try:
        event = sort_prediction_request(request.bitcoin_last_minute)
    except (IndexError, ValueError) as e:
        logger.exception(e)
        return HTTPException(400, detail='Invalid input payload')

    series_data = build_series_df(event)
    transformed_series_data = np.array([scaler.transform(series_data)])

    # The reshape param is simply the length of the input values
    prediction = model.predict(transformed_series_data)
    for_transform = [[0, prediction[0][0], 0, 0, 0, 0]]
    res = scaler.inverse_transform(for_transform)
    return {
        'bitcoin_prediction': round(float(res[0][1]), 2)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
