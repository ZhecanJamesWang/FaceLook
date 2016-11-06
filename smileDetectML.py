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
from sklearn.decomposition import PCA


class SmileDetectML(object):
    def __init__(self):         
        self.clf = SVC(kernel='rbf', C = 1, gamma =0.001)

        self.posDir = "./dataBase/gi/smiling/"
        self.negaDir = "./dataBase/gi/noSmiling/"

        self.posFiles = os.listdir(self.posDir)
        self.negaFiles = os.listdir(self.negaDir)

        self.processDataDir = "./processData/"
        self.debug = False


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
                    # normalize each picture by centering brightness
                    # img -= img.mean(axis=1)[:, np.newaxis]

                    gray, detected_faces = self.detect_face(img)
                    # for (x,y,w,h) in detected_faces:
                    #     cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
                    if self.debug:
                        cv2.imshow('frame1',img)
                    extracted_face = self.extract_face_features(gray, detected_faces[0])
                    if self.debug:
                        cv2.imshow('frame2', extracted_face)
                        k = cv2.waitKey(100) 
                    x.append(extracted_face.ravel())
                    y.append(label)
                    counter += 1
                    if counter %100 == 0:
                        print counter


                except Exception as e:
                    print "."
                    # print detected_faces
                    # cv2.imshow('frame1',img)
                    # cv2.waitKey(0)
                    # raise "debug"
        x = np.asarray(x)
        y = np.asarray(y)

        print x.shape
        print y.shape
        print np.mean(y)

        return x, y

    def decompose(self, xTrain, xTest):
        ################################################################################
        # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
        # dataset): unsupervised feature extraction / dimensionality reduction
        print "xTrain.shape ", xTrain.shape
 
        n_components = 150

        print "Extracting the top %d eigenfaces" % n_components
        pca = PCA(n_components=n_components, whiten=True).fit(xTrain)

        eigenfaces = pca.components_.T.reshape((n_components, 64, 32))

        # project the input data on the eigenfaces orthonormal basis
        xTrainPCA= pca.transform(xTrain)
        xTestPCA = pca.transform(xTest)

        print "xTrainPCA.shape ", xTrainPCA.shape

        return xTrainPCA, xTestPCA

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
        
        # xTrain, xTest = self.decompose(xTrain, xTest)  

        print np.mean(yTrain)
        print np.mean(yTest)

        pickle.dump( xTrain, open( self.processDataDir + "xTrain.p", "wb" ) )
        pickle.dump( yTrain, open( self.processDataDir + "yTrain.p", "wb" ) )
        pickle.dump( xTest, open( self.processDataDir + "xTest.p", "wb" ) )
        pickle.dump( yTest, open( self.processDataDir + "yTest.p", "wb" ) )

    def train(self):
        xTrain = pickle.load( open( self.processDataDir + "xTrain.p", "rb" ) )
        yTrain = pickle.load( open( self.processDataDir + "yTrain.p", "rb" ) )
        xTest = pickle.load( open( self.processDataDir + "xTest.p", "rb" ) )
        yTest = pickle.load( open( self.processDataDir + "yTest.p", "rb" ) )
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


    def extract_face_features(self, gray, detected_face):
        (x, y, w, h) = detected_face

        extracted_face = gray[y+h/2:y+h, 
                          x:x+w]
        new_extracted_face = self.resize(extracted_face, (64, 32))

        return new_extracted_face

    def run(self):
        # self.preProcess()
        self.train()
        # self.save()

if __name__ == '__main__':
    SmileDetectML().run()


