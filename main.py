from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

from model import model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Upload Image Detection
# -----------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)
    return {"image": base64.b64encode(buffer).decode()}


# -----------------------
# Realtime Camera (WebSocket)
# -----------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_bytes()

        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img)
        annotated = results[0].plot()

        _, buffer = cv2.imencode(".jpg", annotated)
        encoded = base64.b64encode(buffer).decode()

        await websocket.send_text(encoded)