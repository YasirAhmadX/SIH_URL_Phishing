from flask import Flask,render_template,request
from decision import modifys, fval 
import pickle
import csv
import numpy as np
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/another_page')
def another_page():
    return render_template('another_page.html')


@app.route('/predict#search-bar-zone',methods=['POST'])
def predict():
    model =pickle.load(open('models/model.pkl','rb'))
    ip = request.form['url']
    val=modifys(ip)
    response = model.predict([val])
    Gflag=fval(ip)
    print(Gflag)
    print(response)
    
    if Gflag:
        if response==1:
            
            return render_template('index.html', prediction_text='Phishing URL Alert!!!',ip=ip)
        else:
            #website not listed in google but seems genuine
            return render_template('index.html', prediction_text='website not listed in google but seems genuine',ip=ip)
        
    else:
        if response == 1:
            return render_template('index.html', prediction_text="Website is listed on google search index, but seems suspicious!",ip=ip)
        else:
            return render_template('index.html', prediction_text='Genuine URL, Proceed...',ip=ip)
        
@app.route('/add',methods=['POST'])
def add():
    ip2=request.form['add']
    val=modifys(ip2)
    val.insert(11,0)
    val.insert(12,0)
    a=val.pop()
    val.append(1)
    val.append(a)
    with open('models\\newdataset.csv','a',newline="") as file:
        writer=csv.writer(file)
        writer.writerow(val)
        return render_template('index.html', addition_text="Successfully added",ip2=ip2)


if __name__ == '__main__':
    app.run(debug=True)