from flask import Flask, render_template, Response,request
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from tensorflow import keras


mp_pose = mp.solutions.pose
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,min_tracking_confidence=0.7)

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

def detectPose(image_pose, pose, draw=False, display=False):
    
    original_image = image_pose.copy()
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    resultant = pose.process(image_in_RGB)
    if resultant.pose_landmarks and draw:    
        mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                               thickness=2, circle_radius=2))

    if display:
            
            plt.figure(figsize=[22,22])
            plt.subplot(121);plt.imshow(image_pose[:,:,::-1]);plt.title("Input Image");plt.axis('off');
            plt.subplot(122);plt.imshow(original_image[:,:,::-1]);plt.title("Pose detected Image");plt.axis('off');
    
    else:
        pose_ = np.array(([[res.x,res.y,res.z,res.visibility] for res in resultant.pose_landmarks.landmark] )).flatten() if resultant.pose_landmarks else np.zeros(1404) 
        return np.concatenate([pose_])

model = keras.models.load_model('Pose_detection.h5')

inverted_labels = {'downdog': 0, 'goddess': 1, 'plank': 2, 'tree': 3, 'warrior2': 4}
labels = {}
for i in inverted_labels.items():
    labels[i[1]] = i[0]
labels

def image(typ):
        print("------------>test",typ)
        output = cv2.imread(typ)
        cc = detectPose(output, pose_image, draw=False, display=False)
        lab = labels[np.argmax(model.predict(cc.reshape(-1,132)))]
        cv2.rectangle(output,(0,0),(200,40),(255,255,0),-1)
        cv2.putText(output,lab,(3,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
        #cv2.imshow("test",output)
        #cv2.imwrite("last.jpg", output)
        ret, buffer = cv2.imencode('.jpg', output)
        frame = buffer.tobytes()
        # generator - yield
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def live(typ):
    cap = cv2.VideoCapture(int(typ))
    while True: 
        _,frame = cap.read()
        if _:
            pose_points = detectPose(frame, pose_video, draw=True, display=False)
            try:
                label = labels[np.argmax(model.predict(pose_points.reshape(-1,132)))]
            except:
                pose_points = np.ones([1,132])
                continue
            cv2.rectangle(frame,(0,0),(640,40),(255,0,0),-1)
            cv2.putText(frame,label,(3,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            #cv2.imshow('Pose Detection Project TYCO', frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')







app = Flask(__name__)
#run_with_ngrok(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lives')
def lives():
    path = 0
    return Response(live(path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/images', methods=['POST'])
def images():
    file = request.files['Image']
    file.save('static/' + file.filename)  # Save the uploaded file to a folder called 'static'
    url = 'static/' + file.filename
    return Response(image(url), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return render_template('display.html', video_url=video_url)


if __name__ == "__main__":
    app.run(debug=True)