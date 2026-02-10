import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn


# Load All Models
haar = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
# pickle files
mean = pickle.load(open('model/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open('model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('model/pca_50.pickle', 'rb'))
print('Model Loaded Successfully')

gender_pre = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX


# Piepline Function
def pipeline_model(img_path, color='rgb'):
    img = cv2.imread(img_path)
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = haar.detectMultiScale(gray, 1.1, 6)
    
    predictions = []   # 🔥 list of dictionaries

    eigen_image = None

    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        roi = gray[y:y+h, x:x+w]
        roi = roi/255

        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (244,244), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (244,244), cv2.INTER_CUBIC)

        roi_reshape = roi_resize.reshape(1,59536)
        roi_mean = roi_reshape - mean
        eigen_image = model_pca.transform(roi_mean)
        results = model_svm.predict_proba(eigen_image)[0]

        predict = results.argmax()
        score = results[predict]

        label = gender_pre[predict]
        text = "%s : %0.2f"%(label, score)

        # 🎨 Colors (BGR format)
        if label == "Male":
            color = (255, 0, 0)      # Blue
        else:
            color = (203, 192, 255)  # Pink (light)

        # Rectangle
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)

        # Text background
        cv2.rectangle(img, (x, y-60), (x+w, y), color, -1)

        # Text
        cv2.putText(img, text, (x+5, y-10), font, 2.0, (0,0,0), 5)

        # ========== Save Prediction Object ==========
        pred_obj = {
            'roi' : roi_resize,                                                                    # gray roi
            'eig_img' : model_pca.inverse_transform(eigen_image[0]).reshape(244,244),              # eigen vector
            'prediction_name' : label,                                                             # Male/Female
            'score' : score                                                                        # confidence
        }

        predictions.append(pred_obj)

    return img, predictions



