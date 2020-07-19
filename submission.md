# Execution
* App is developed using FastAPI.
* to quick deploy, use docker compose ```docker-compose up```
* app can also be run using standard docker commands :
    * ```docker build -t penn_app:latest .```
    * ```docker run -p 8000:8000```
* Navigate to http://localhost:8000 (or http://localhost:8000/health) to ensure app is running

# Answers to Questions

In the spirit of your instructions, I only took a few hours to complete the challenge. So rather then implement everything, I wanted to discuss what I would of done for some of these iterations.

Q: How do we write good tests (unit, integration) for this machine learning service? \
A: There are multiple good ways to test the service itself.  
* FastAPI provides a TestClient allowing for easy test writing for the API endpoints. An example test for our root endpoint is shown below. You can use the same approach to test all endpits, ensuring to test all success AND failure cases.
~~~~{.python}
from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello Penn"}
~~~~
 * We can also write unit tests for any helper functions in our ap using the standard unittest framework.
 * We can also use POSTMAN scripts to automate endpoint testing.    
\
Q: Can we add an endpoint just to check that the API is up? \ 
A: You bet we can, the answer is health checks! (see code) \
\
Q Tensorflow throws a lot of Future warnings and CPU warnings at runtime. Can we fix the code so these warnings go away? \
A: Depending on your definition of fix... Adding this line to the imports of our app will mute all tensorflow logs except errors
~~~~{.python}
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
~~~~
\
Q: Is the neural network architecture correct? / Is a different model appropriate? \
A: Because Bitcoin, like regular stock-market data, is time-series data, a more appropriate network architecture for 
forecasting the price of Bitcoin at time t+1 are recurrent neural networks. In particular, RNNs with LSTM units to assist with vanishing gradients. GRUs could also be used if a speed up in training time is desired, less it drastically effect model correctness. (I have made these changes in the Jupyter Notebook)\
\
Q: Does accuracy make sense in this context? \ 
A: No. This problem is a regression problem, not a classification one. So accuracy does not make sense. Traditional regression statistics like MSE and RMSE make sense for validation metrics here. \
\
Q: Do we have the right feature set? More features? Less features? Better features? \
A: I think the better feature set is to take the data as is and split it into multiple time series with 59 observations and 1 price_high observation as the target variable.I had to change from 60 to 59 because the incoming request has data for time 0 with no price_high. With the new model architecture, we do not need that tme 0 row. \
\
Q:Is there any scaling or preprocessing we're missing?
Scaling is often a good idea. Here, because the distance between our features is quite high, using a scalar such as MinMax wil allow the network to better represent relationships between our features. \ 
\
Q: What version is the API on and how do we know? Is this the same as the model version? Do we need both? \ 
A: We can specify versions as an environment variables in the application. We can also add an endpoint to retrieve them. The application and model versions should be independent. An increment to the application version should represent new features or bug fixes, an increment to the model should be from a new model being used by the application. These should be de-coupled. 
 
