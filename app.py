from models import *
from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import datetime
import time

app = Flask(__name__)

var_list = []


def capture_by_frames():
    global camera
    camera = cv2.VideoCapture(0)
    sampleNum = 0
    d = var_list.pop()
    name = str(d[2])
    Id = str(d[0])

    while True:
        success, frame = camera.read()
        detector = cv2.CascadeClassifier(
            'Haarcascades/haarcascade_frontalface_default.xml')
        faces = detector.detectMultiScale(frame, 1.3, 5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in faces:
            x1, y1 = x+w, y+h
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 6)
            cv2.line(frame, (x, y), (x+30, y), (0, 255, 255), 3)
            cv2.line(frame, (x, y), (x, y+30), (0, 255, 255), 3)

            cv2.line(frame, (x1, y), (x1-30, y), (0, 255, 255), 3)
            cv2.line(frame, (x1, y), (x1, y+30), (0, 255, 255), 3)

            cv2.line(frame, (x, y1), (x+30, y1),
                     (0, 255, 255), 3)
            cv2.line(frame, (x, y1), (x, y1-30), (0, 255, 255), 3)

            cv2.line(frame, (x1, y1), (x1-30, y1),
                     (0, 255, 255), 3)
            cv2.line(frame, (x1, y1), (x1, y1-30), (0, 255, 255), 3)
            sampleNum = sampleNum+1
            cv2.imwrite("Image/ "+name + "." + Id + '.' + str(sampleNum) +
                        ".jpg", gray[y:y+h, x:x+w])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif sampleNum > 20:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    writeCSV(d)


def detect_capture_by_frames():
    global camera
    camera = cv2.VideoCapture(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Training/Trainner.yml")
    detector = cv2.CascadeClassifier(
        'Haarcascades/haarcascade_frontalface_default.xml')
    df = pd.read_csv("data/list.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        success, frame = camera.read()

        faces = detector.detectMultiScale(frame, 1.3, 5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in faces:
            x1, y1 = x+w, y+h
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.line(frame, (x, y), (x+30, y), (255, 0, 0), 6)
            cv2.line(frame, (x, y), (x, y+30), (255, 0, 0), 6)

            cv2.line(frame, (x1, y), (x1-30, y), (255, 0, 0), 6)
            cv2.line(frame, (x1, y), (x1, y+30), (255, 0, 0), 6)

            cv2.line(frame, (x, y1), (x+30, y1),
                     (255, 0, 0), 6)
            cv2.line(frame, (x, y1), (x, y1-30), (255, 0, 0), 6)

            cv2.line(frame, (x1, y1), (x1-30, y1),
                     (255, 0, 0), 6)
            cv2.line(frame, (x1, y1), (x1, y1-30), (255, 0, 0), 6)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(
                    ts).strftime('%H:%M:%S')
                cv2.putText(frame, str(Id), (x, y+h),
                            font, 1, (255, 255, 255), 2)

            else:
                Id = '??????'
                cv2.putText(frame, str(Id), (x, y+h),
                            font, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/takePhotos", methods=['POST'])
def takePhotos():
    number = request.form.get('number')
    Name = request.form.get('Name')
    idName = request.form.get('idName')
    arrList = [number, Name, idName]
    var_list.append(arrList)
    return render_template('index.html')


@app.route("/")
def Lists():
    result = readCSV()
    return render_template('list.html', data=result)


@app.route('/add')
def add():
    return render_template('add.html')


@app.route('/start', methods=['POST'])
def start():
    return render_template('add.html')


@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/stop', methods=['POST'])
def stop():
    if camera.isOpened():
        camera.release()
    return render_template('stop.html')


@app.route('/stopDetect', methods=['POST'])
def stopDetect():
    if camera.isOpened():
        camera.release()
    return render_template('list.html')


@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_video_capture')
def detect_video_capture():
    return Response(detect_capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/TrainImage', methods=['POST'])
def TrainImage():
    faces, Id = getImagesAndLabels('image')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(
        'Haarcascades/haarcascade_frontalface_default.xml')
    recognizer.train(faces, np.array(Id))
    recognizer.save("Training/Trainner.yml")
    return Lists()


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


if __name__ == '__main__':
    app.run(debug=True)
