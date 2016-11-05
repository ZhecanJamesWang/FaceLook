from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import os
from PIL import Image, ImageChops
import cv2
import numpy as np
import pickle
from random import shuffle
from sklearn.externals import joblib
from scipy.ndimage import zoom

class Model(object):
    def __init__(self):         
        self.clf = SVC(kernel='rbf')

        self.posDir = "./dataBase/gi/smiling/"
        self.negaDir = "./dataBase/gi/noSmiling/"

        self.posFiles = os.listdir(self.posDir)
        self.negaFiles = os.listdir(self.negaDir)


    def resize(self, image, size):

        image = Image.fromarray(np.uint8(image))
        image.thumbnail(size, Image.ANTIALIAS)
        image_size = image.size

        thumb = image.crop( (0, 0, size[0], size[1]) )

        offset_x = max( (size[0] - image_size[0]) / 2, 0 )
        offset_y = max( (size[1] - image_size[1]) / 2, 0 )

        thumb = ImageChops.offset(thumb, offset_x, offset_y)
        image = np.asarray(thumb)
        return image


    def generateData(self, files, directory, label):
        x = []
        y = []
        counter = 0
        for file in files:
            if file != "DS_Store":
                try:


                    img = cv2.imread( directory + file, 1)
                    gray, detected_faces = self.detect_face(img)
                    # print detected_faces[0]
                    # raise "debug"
                    # for (x,y,w,h) in detected_faces:
                    #     cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
                    cv2.imshow('frame111',img)
                    # k = cv2.waitKey(10)  
                    # raise "debug"   
                    # for face in detected_faces:                
                    extracted_face = self.extract_face_features(gray, detected_faces[0], (0.03, 0.05))
                    # print extracted_face
                    cv2.imshow('frame', extracted_face)
                    k = cv2.waitKey(100) 
                    # img = self.resize(img, (64, 64))
                    x.append(extracted_face.ravel())
                    y.append(label)
                    counter += 1
                    print counter


                except Exception as e:
                    print e
        x = np.asarray(x)
        y = np.asarray(y)

        print x.shape
        print y.shape
        print np.mean(y)

        return x, y

    def preProcess(self):
        xPos, yPos = self.generateData(self.posFiles, self.posDir, 1)
        xNega, yNega = self.generateData(self.negaFiles, self.negaDir, 0)

        x, y = [], []
        x.extend(xPos) 
        x.extend(xNega)

        y.extend(yPos) 
        y.extend(yNega)


        x = np.asarray(x)
        y = np.asarray(y)

        shuffle(x)
        shuffle(y)

        print "x.shape ", x.shape
        print "y.shape ", y.shape    
        
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25, random_state=0)
            
        print np.mean(yTrain)
        print np.mean(yTest)

        # pickle.dump( xTrain, open( "xTrain.p", "wb" ) )
        # pickle.dump( yTrain, open( "yTrain.p", "wb" ) )
        # pickle.dump( xTest, open( "xTest.p", "wb" ) )
        # pickle.dump( yTest, open( "yTest.p", "wb" ) )

    def train(self):
        xTrain = pickle.load( open( "xTrain.p", "rb" ) )
        yTrain = pickle.load( open( "yTrain.p", "rb" ) )
        xTest = pickle.load( open( "xTest.p", "rb" ) )
        yTest = pickle.load( open( "yTest.p", "rb" ) )
        print xTrain.shape
        print yTrain.shape
        print xTest.shape
        print yTest.shape


        self.clf.fit(xTrain, yTrain)

        print ("Accuracy on training set:")
        print (self.clf.score(xTrain, yTrain))
        print ("Accuracy on testing set:")
        print (self.clf.score(xTest, yTest))

    def save(self):
        joblib.dump(self.clf, 'faceDetectSVC.pkl') 

    def load(self):
        svc_1 = joblib.load('faceDetectSVC.pkl') 

    def detect_face(self, frame):
        cascPath = './lib/haarcascade_frontalface_alt.xml'
        faceCascade = cv2.CascadeClassifier(cascPath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(100, 100),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        return gray, detected_faces


    def extract_face_features(self, gray, detected_face, offset_coefficients):
        (x, y, w, h) = detected_face

        horizontal_offset = offset_coefficients[0] * w
        vertical_offset = offset_coefficients[1] * h

        extracted_face = gray[y+vertical_offset+h/2:y+h, 
                          x+horizontal_offset:x-horizontal_offset+w]

        new_extracted_face = zoom(extracted_face, (32. / extracted_face.shape[0], 
                                               64. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)

        new_extracted_face /= float(new_extracted_face.max())

        return new_extracted_face

    def run(self):
        self.preProcess()
        # self.train()
        # self.save()

if __name__ == '__main__':
    Model().run()


