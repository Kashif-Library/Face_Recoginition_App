from flask import render_template, request
import os
import cv2
from app.face_recoginition import pipeline_model 
import matplotlib.pyplot as plt

UPLOAD_FOLDER = "static/upload"

def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def genderapp():
    
    if request.method == "POST":
        f = request.files['image-name']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        
        pred_image, predictions = pipeline_model(path)
        
        pred_filename = "Prediction_Image.jpg"
        
        cv2.imwrite(f"./static/predict/{pred_filename}", pred_image)
        print(predictions)
        
        report = []
        
        for i,obj in enumerate(predictions):
            gray_img = obj['roi']
            eigen_img = obj['eig_img']
            gender_name = obj['prediction_name']
            score = round(obj['score']*100, 2)
            
            gray_img_name = f'roi_{i}.jpg'
            eig_img_name = f'eigen_{i}.jpg' 
            
            plt.imsave(f'./static/predict/{gray_img_name}', gray_img, cmap='gray')
            plt.imsave(f'./static/predict/{eig_img_name}', eigen_img, cmap='gray')
            
            report.append([gray_img_name, eig_img_name, gender_name, score])
            
        return render_template('gender.html', fileupload=True)
        
    print("Machine Learning Model Executed Successfully")
    
    return render_template('gender.html', fileupload=False)