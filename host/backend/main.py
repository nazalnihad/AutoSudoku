from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import numpy as np
import io

from utils import setModel, preProcess, largestContour, reorder, splitImg, getPredictions

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI()

# Allowing CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    numb_board = process_and_extract_board(contents)
    if numb_board is not None:
        return numb_board
    else:
        return {"error": "No Sudoku board found in the image."}

def process_and_extract_board(image_data):
    img_height = 450
    img_width = 450
    model = setModel()

    # Convert image data to OpenCV format
    img = np.array(Image.open(io.BytesIO(image_data)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (img_width, img_height))
    img_threshold = preProcess(img)

    # Find contours
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest, max_area = largestContour(contours)

    if largest.size != 0:
        largest = reorder(largest)
        pts1 = np.float32(largest)
        pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarp = cv2.warpPerspective(img, matrix, (img_width, img_height))
        imgWarp = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)

        # Split image into boxes and get predictions
        boxes = splitImg(imgWarp)
        numbers = getPredictions(boxes, model)
        numbers = np.asarray(numbers)
        board = np.array_split(numbers, 9)

        # Flatten the board
        numb_board = np.reshape(numbers, (9, 9))
        return numb_board.tolist()  # Convert to a list for JSON serialization
    else:
        return None