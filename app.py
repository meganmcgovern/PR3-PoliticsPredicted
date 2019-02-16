import os

import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['DEBUG'] = True

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

@app.route("/party-prediction")
def element_correlation():
    return render_template("party-prediction.html")

@app.route("/twitter")
def timeseries():
    return render_template("twitter.html")


if __name__ == "__main__":
    app.run()
