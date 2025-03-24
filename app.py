from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from disease_prediction import pre_process
from disease_prediction.config import UPLOAD_FOLDER, allowed_file
from crop_prediction import crop_pre_process

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# website home page
@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

# crop prediction page

@app.route("/crop", methods=['GET','POST'])
def crop_model():
    if request.method=="GET":
        return render_template('crop_prediction.html')
    
    else:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall =float(request.form['rainfall'])
        result=crop_pre_process.crop_model(N,P,K,temperature,humidity,ph,rainfall)
        print(result)
        
        
    return render_template('crop_prediction.html',result1=result)


#disease prediction page

@app.route("/disease", methods=["GET", "POST"])
def disease_model():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('disease_prediction.html', error='No file part')
            
        file = request.files['file']
        
        if file.filename == '':
            return render_template('disease_prediction.html', error='No selected file')
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get the relative path for the template
            image_path = url_for('static', filename='uploads/' + filename)
            
            result = pre_process.model_train(filepath)
            return render_template('disease_prediction.html', result=result, image_path=image_path)
            
    return render_template('disease_prediction.html')


# water management page 
@app.route("/water", methods=['GET','POST'])
def water_model():
    if request.method=="GET":
        return render_template('water_management.html')
    
    else:
        sc=float(request.form['maths'])
        mh=float(request.form['science'])
        avgs=(sc+mh)/2
        return render_template('water_management.html',score=avgs)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
