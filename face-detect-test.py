
import cv2
import sys
import os


class FaceCropper(object):
    CASCADE_PATH = "api/utils/haar_classifiers/face_detect_model.xml"

    def __init__(self):
        print("initialized Cascade Classifier")
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("detect face")
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return 0

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)

        if (show_result):
            print("Show result")
            for (x, y, w, h) in faces:
                # x = x + 50
                # y = y + 100
                # w = w
                # h = h + 20
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        i = 0
        height, width = img.shape[:2]

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (32, 32))
            i += 1
            cv2.imwrite("image%d.jpg" % i, lastimg)


detecter = FaceCropper()
detecter.generate('tests/test_images/test5.jpg', True)

