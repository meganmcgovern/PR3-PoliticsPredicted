import os

import pandas as pd

import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from pyspark.sql import SparkSession
from pyspark import SparkFiles
spark = SparkSession.builder.appName('prez').getOrCreate()
from pyspark.ml.classification import NaiveBayesModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.sql.functions import length
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG'] = True

model = NaiveBayesModel.load('models/naivebayes.h5')

def pipeline(df):
    print(df.head())
    df = df.withColumn("length", length(df['Speech']))
    # Create the data processing pipeline functions here (note: StringIndexer will be used to encode 
    # your target variable column. This column should be named 'label' so our model will recognize it later)
    review_data = Tokenizer(inputCol="Speech", outputCol="Words")
    reviewed = review_data.transform(df)
    #reviewed.show()
    remover = StopWordsRemover(inputCol="Words", outputCol="filtered")
    newFrame = remover.transform(reviewed)
    #newFrame.show()
    hashing = HashingTF(inputCol="filtered", outputCol="hashedValues", numFeatures=pow(2,10))
    # Transform in a DF
    hashed_df = hashing.transform(newFrame)
    hashed_df.show(truncate=False)
    idf = IDF(inputCol="hashedValues", outputCol="feature")
    idfModel = idf.fit(hashed_df)
    rescaledData = idfModel.transform(hashed_df)
    rescaledData.select("words", "feature").show(truncate=False)
    # indexer = StringIndexer(inputCol="Party_Affliation", outputCol="label")
    
    # indexed = indexer.fit(rescaledData).transform(rescaledData)
    
    assembler = VectorAssembler(
    inputCols=["feature", "length"],
    outputCol="features")

    return assembler.transform(rescaledData)


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

@app.route("/party-prediction-input", methods=['GET', 'POST'])
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
            cleaned = pipeline(spark.read.format("csv").option("header", "true").load("speech.csv"))


            # use model to clean up 'speech.csv' i.e. cleaned = (cleaned csv)

            # Use the model to make a prediction
            predictions = model.transform(cleaned)

            data["predictions"] = str(predictions.select('prediction').collect()[0])
            data['predictions'] = 'Democrat' if data['predictions'] == 'Row(prediction=1.0)' else 'Republican'

            # indicate that the request was a success
            data["success"] = True

            return render_template('response.html', data=data)

    return render_template("party-prediction-input.html")




if __name__ == "__main__":
    app.run()
