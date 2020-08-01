from flask import Flask, request, render_template
from keras.models import load_model
import pickle
global model, graph
import tensorflow as tf
graph = tf.get_default_graph()

with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)

model = load_model('mymodel.h5')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyse', methods = ['POST'])
def review():
    input_text = request.form['a']
    
    with graph.as_default():
        x_intent = cv.transform([input_text])
        y_pred = model.predict(x_intent)
        out = ''
        if(y_pred>0.5):
            out = "Positive Review"
        else:
            out = "Negative Review"
        
        return render_template('index.html', review=out)