from flask import Flask,render_template,url_for,request,jsonify
import joblib
import pandas as pd
model=joblib.load('km_model.lb')
std = joblib.load('standard_scalar.lb')
df = pd.read_csv('predictin_label.csv')



app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input_data')
def input_data():
    return render_template('input_data.html')

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        Nitrogen=request.form['nitrogen']
        Phosphorus=request.form['phosphorus']
        Potassium=request.form['potassium']
        Temperature=request.form['temprature']
        Humidity=request.form['humidity']
        pH=request.form['ph']
        Rainfall=request.form['rainfall']
        x=[Nitrogen,Phosphorus,Potassium,Temperature,Humidity,pH,Rainfall]
        x1=[x]
        x_trans = std.fit_transform(x1)
        pred = model.predict(x1)[0]
        gr = df[df['group_22']==pred]
        out=gr['label'].value_counts().keys()
        lst = list(out)
        return render_template('prediction.html',output=str())



if __name__ == "__main__":
    app.run(debug=True)