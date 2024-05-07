from utils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img_path = 'resources/sudoku_img_test.png'
img_height = 450
img_width = 450
model = setModel()

# img prep
img = cv2.imread(img_path)
img = cv2.resize(img,(img_width,img_height))
img_threshold = preProcess(img)
# cv2.imshow('image',img_threshold)
# cv2.waitKey(0)

# contour or margin finding
imgContours = img.copy()
imgLargeContours = img.copy()
contours , hierarchy = cv2.findContours(img_threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),3)
# cv2.imshow('image',imgContours)
# cv2.waitKey(0)

# largest contour
largest,max_area = largestContour(contours)
print(largest)

if largest.size !=0 :
    largest = reorder(largest)
    print(largest)
    cv2.drawContours(imgLargeContours,largest,-1,(0,255,0),25)
    pts1 = np.float32(largest) 
    pts2 = np.float32([[0,0],[img_width,0],[0,img_height],[img_width,img_height]])
    marix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,marix,(img_width,img_height))
    imgWarp = cv2.cvtColor(imgWarp,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image',imgWarp)
    # cv2.waitKey(0)

# sudoku board
    imgSolvedDigits = imgWarp.copy()
    boxes = splitImg(imgWarp)
    print(len(boxes))
    numbers = getPredictions(boxes,model)
    print(numbers)
    numbers = np.asarray(numbers)
    board = np.array_split(numbers,9)
    new_board = np.reshape(numbers,(9,9))
    print(board)
    print(new_board)

# model.summary()