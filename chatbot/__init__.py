import flask
from flask import Flask
from chatbotconfig import Config
# import googlesearch
app=Flask(__name__)
app.config.from_object(Config)

import keras
import nltk
import pickle
import json
from keras.models import load_model

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

model=load_model('model/mymodel.h5')
intents = json.loads(open('intents.json',encoding="utf-8").read())
words = pickle.load(open('model/words.pkl','rb'))
classes = pickle.load(open('model/classes.pkl','rb'))


from chatbot import routes
