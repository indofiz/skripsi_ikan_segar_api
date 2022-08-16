from io import BytesIO

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.models import model_from_json

model = None


def load_model():
    # model = tf.keras.applications.MobileNetV2(weights="imagenet")
    json_file = open('model_selar.json','r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights('selar_como_augmented.h5')
    print("Model loaded")
    return model


# def predict(image: Image.Image):
#     global model
#     if model is None:
#         model = load_model()

#     image = np.asarray(image.resize((224, 224)))[..., :3]
#     image = np.expand_dims(image, 0)
#     image = image / 255.0

#     result = decode_predictions(model.predict(image), 2)[0]

#     response = []
#     for i, res in enumerate(result):
#         resp = {}
#         resp["class"] = res[1]
#         resp["confidence"] = f"{res[2]*100:0.2f} %"

#         response.append(resp)

#     return response

def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()
    shape = 224
    img = np.fromstring(image, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    image_test = cv2.resize(img,(shape,shape))
    image_test = np.reshape(image_test,[1,shape,shape,3])
    prediksi = model.predict(image_test)
    classes_x=np.argmax(prediksi,axis=1) + 1
    hasil = classes_x.tolist()

    return hasil[0]


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image