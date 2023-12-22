from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import model_instance, NgramModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
async def predict(sequence: str, num_predictions: int = 4):
    return model_instance.predict_next_chars(sequence, num_predictions)

@app.get("/predict_no_space")
async def predict_no_space(sequence: str, num_predictions: int = 4):
    return model_instance.predict_next_chars_no_space(sequence, num_predictions)


# In this setup:

# FastAPI() creates a new FastAPI application.
# The @app.get("/predict") decorator defines a route for predictions that include spaces.
# The @app.get("/predict_no_space") decorator defines a route for predictions that exclude spaces.
# Each function takes a sequence and an optional num_predictions parameter from the query and returns the predictions using your N-gram model.