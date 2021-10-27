import urllib.request
import cv2
import numpy as np
import os

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
dataset = 'dataset'
print('\nModel is Training...........')

(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(dataset):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(dataset, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id +=1
        
(images, labels) = [np.array(lis) for lis in [images, labels]]
print((images, labels))
(width, height) = (130, 100)

model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)

cnt = 0

# download IP Webcam App in mobile and paste here the url given in the app 
url='http://192.168.43.1:8080/shot.jpg'

while True:
    # we are reading frames from mobile cam
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img =  cv2.resize(img, None, fx= 0.4, fy= 0.4)
    #cv2.imshow("CameraFeed",img)

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(grayImg, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 1)
        face= grayImg[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
        if prediction[1]<800:
            cv2.putText(img, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10),
            cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))
            
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt +=1
            cv2.putText(img, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
            if cnt>100:
                print('Unknown Person')
                cv2.imwrite('unknown.jpg', img)
                cnt = 0

    cv2.imshow('FaceRecognition', img)
    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()