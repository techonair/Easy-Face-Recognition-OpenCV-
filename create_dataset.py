# Before training model we need to create a dataset contain person and his images 
# before running this code make sure to create dataset folder in the same directory
import cv2, os

haar_file = 'haarcascade_frontalface_default.xml'
dataset = 'dataset'
sub_data = input('Person Name: ')

path = os.path.join(dataset, sub_data)

if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)

count = 0

while count < 31:
    print(count)
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
        face = grayImg[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite("%s/%s.png" % (path, count), face_resize)
    
    count += 1

    cv2.imshow('Capturing', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()