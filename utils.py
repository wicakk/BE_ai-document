import cv2
import numpy as np
from PIL import Image

def predict_image(model, image_bytes):
    img = Image.open(image_bytes).convert("RGB")
    img = np.array(img)

    results = model(img)

    return results[0].plot()  # return annotated image