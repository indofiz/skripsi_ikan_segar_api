from typing import Union
from fastapi import FastAPI, File, UploadFile
from predict import predict, read_imagefile

app = FastAPI()


@app.get("/")
def read_root():
    return {'message': 'Welcome to Ikan Segar UBB'}


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    # image = read_imagefile(await file.read())
    image = await file.read()
    prediction = predict(image)

    return prediction

