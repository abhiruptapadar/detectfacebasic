import cv2
# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
# Read the input image
img = cv2.imread("together.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5)
eyes = eye_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    img=cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
resized=cv2.resize(img,(int(img.shape[1]),int(img.shape[0],)))
cv2.imshow("Gray",resized)

cv2.waitKey(0)
cv2.destroyAllWindows()