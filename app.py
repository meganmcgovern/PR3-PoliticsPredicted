import os

import pandas as pd
import keras
from keras.preprocessing import image
from keras import backend as K

import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG'] = True

def load_model():
    global model
    global graph
    model = keras.models.load_model("mnist_trained.h5")
    graph = K.get_session().graph

load_model()

def clean_user_input(files):  
    with open(files) as file:
        data = file.read() 
        split_speech = data.splitlines(True)
        final_list = []
        for i in split_speech:
            i = i.strip('\n')
            if i == '':
                pass
            else:
                final_list.append(i)
        #print(final_list)
        speech = ' '.join(final_list)
        party = ''
        #print(speech)
        speech = speech.replace('.', ' ')
        speech = speech.replace(',', ' ')
        speech = speech.replace('“', ' ')
        speech = speech.replace('”', ' ')
        speech = speech.replace(':', ' ')
        speech = speech.replace('’', ' ')
        speech = speech.replace('$', ' ')
        speech = speech.replace('—', ' ')
        print(len(speech))

    df = pd.DataFrame({'Speech': speech}, index=[0])
    df.to_csv('speech.csv', sep=',', index=False)

# Database Setup
#################################################

# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db/##.sqlite"
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# Base = automap_base()
# Base.prepare(db.engine, reflect=True)

# twitter_politics = Base.classes.twitter_tweet_data

# @app.route("/twitter-data")
# def twitter_data():
#     stmt = db.session.query(twitter_politics).statement
#     df = pd.read_sql_query(stmt, db.session.bind)

#     return df.to_json()

@app.route("/")

def index():
    return render_template("index.html")

@app.route("/party-prediction", methods=['GET', 'POST'])
def element_correlation():
    data = {"success": False}
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)
            clean_user_input(filepath)

            # use model to clean up 'speech.csv' i.e. cleaned = (cleaned csv)

            # global graph
            # with graph.as_default():

            #     # Use the model to make a prediction
            #     prediction_digit = model.predict_classes(cleaned)
            #     data["prediction"] = str(prediction)

            #     # indicate that the request was a success
            #     data["success"] = True

            return jsonify(data)

    return render_template("party-prediction.html")

@app.route("/twitter")
def timeseries():
    return render_template("twitter.html")


if __name__ == "__main__":
    app.run()
