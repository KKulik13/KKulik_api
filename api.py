from flask import Flask
from flask_restful import Resource, Api
import cv2


app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('images/dworzec.jpeg')
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        # print(type(img))
        # print(img.shape)
        # print(len(boxes))
        return {'count': len(boxes)}

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class HelloWorld2(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(HelloWorld, '/test')
api.add_resource(HelloWorld2, '/test2')
api.add_resource(PeopleCounter, '/')

if __name__ == '__main__':
    app.run(debug=True)
