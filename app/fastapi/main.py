from fastapi import FastAPI
import pickle
from mangum import Mangum
import uvicorn
import contextlib
from data_model import CropGroupsModel, PredictionInput, PredictionOutput
from fastapi import Depends, FastAPI

app = FastAPI()


crop_groups_model = CropGroupsModel()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    crop_groups_model.load_model()
    yield


app = FastAPI(lifespan=lifespan)



@app.post("/prediction")
async def prediction(input: PredictionInput) -> PredictionOutput:
    output = crop_groups_model.predict(input)
    # print(output)
    return output

handler = Mangum(app)



if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)