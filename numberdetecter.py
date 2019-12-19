import cv2
import time
from keras.models import load_model

model=load_model('mnist_trained_model.h5')

video = cv2.VideoCapture(0)
video.read()
check, frame = video.read()
#frame=cv2.imread("test.png",1)
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(21,21),0)

cv2.imwrite("picture.png", frame)
cv2.imwrite("graypic.png",gray)

check,thresholdimg=cv2.threshold(gray,107,255,cv2.THRESH_BINARY)
cv2.imwrite("threshold.png",thresholdimg)

check,threholdinginvt=cv2.threshold(gray,107,255,cv2.THRESH_BINARY_INV)
cv2.imwrite("thresholdinvt.png",threholdinginvt)

resized=cv2.resize(threholdinginvt,(28,28),interpolation=cv2.INTER_AREA)
cv2.imwrite("resized.png",resized)
im_final = resized.reshape(1,28,28,1)
ans=model.predict(im_final)
print(ans)

cv2.destroyAllWindows()
