
import os
from os import path

import numpy as np

from wordcloud import WordCloud, STOPWORDS

import nltk
from nltk import sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import PIL
from PIL import Image, ImageDraw, ImageFont

import matplotlib

sid = SentimentIntensityAnalyzer()



stopwords = set(STOPWORDS)


#Text Pre Processing
def review_tokenize(review_list):
  filtered_text = ''
  
  for review in review_list:
    sentences = nltk.sent_tokenize(review)
    for sentence in sentences:
      tokens = nltk.pos_tag(nltk.word_tokenize(sentence))

      for i in tokens:
        if i[1] == "JJ":
          filtered_text += i[0] + " "
  return filtered_text

#Color Reviews based on sentiment
def green_red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl({}, 100%, 50%)".format(int(70.0 * sid.polarity_scores(word)["compound"] + 45.0))





#Create a wordcloud
def create_wordcloud(text, name):
  mask = np.array(PIL.Image.open("var/www/FlaskApp/static/Black_Circle.jpg").resize((540,540)))
  wc = WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGBA", max_words=1000, mask=mask, stopwords=stopwords, margin=5,
               random_state=1).generate(text)
  wc.recolor(color_func=green_red_color_func)
  wc.to_file("/var/www/FlaskApp/FlaskApp/" + name + ".png")
