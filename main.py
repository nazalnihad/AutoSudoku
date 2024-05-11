from utils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def process_image(img_path):
    img_height = 450
    img_width = 450
    model = setModel()

    # Read and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))
    img_threshold = preProcess(img)

    # Find contours
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest, max_area = largestContour(contours)

    if largest.size != 0:
        largest = reorder(largest)
        pts1 = np.float32(largest) 
        pts2 = np.float32([[0,0],[img_width,0],[0,img_height],[img_width,img_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarp = cv2.warpPerspective(img, matrix, (img_width, img_height))
        imgWarp = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)

        # Split image into boxes and get predictions
        boxes = splitImg(imgWarp)
        numbers = getPredictions(boxes, model)
        numbers = np.asarray(numbers)
        board = np.array_split(numbers, 9)
        # Flatten the board
        numb_board = np.reshape(numbers,(9, 9))

        return numb_board
    else:
        return None

# Example usage
img_path = 'resources/test5.jpg'
board = process_image(img_path)
if board is not None:
    print(board)
else:
    print("No Sudoku board found in the image.")
