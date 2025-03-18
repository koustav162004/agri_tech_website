from flask import Flask,render_template,request,redirect,url_for
import os
from werkzeug.utils import secure_filename
import pre_process
from config import UPLOAD_FOLDER, allowed_file  #



app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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
        sc=float(request.form['maths'])
        mh=float(request.form['science'])
        avgs=(sc+mh)/2
        return render_template('crop_prediction.html',score1=avgs)


#disease prediction page
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  

@app.route("/disease", methods=["GET", "POST"])
def disease_model():
    result = None  # Default result message

    if request.method == "GET":
        return render_template("disease_prediction.html", result=result)
    
    else:


        file = request.files.get("file")  # Get file

        if not file or file.filename == "":
            result = "No file selected. Please upload an image."
        elif not allowed_file(file.filename):
            result = "Invalid file type. Please upload an image (jpg, png, etc.)."
        else:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image with your model
            result = pre_process.model_train(filepath)

    return render_template("disease_prediction.html", result=result)

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
    app.run(debug=True)
