
import sys
import os
import json
from json import JSONEncoder


"""
Entry Point for Mota Analytica Flask Application
"""
#Dependencies
#   NLTK
import nltk
from nltk import sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#   Flask
from flask import Flask, request, jsonify,render_template, redirect, url_for
from flask_pymongo import PyMongo
from flask_cors import CORS

#WordCloud
from FlaskApp.clouds import review_tokenize, green_red_color_func, create_wordcloud
from FlaskApp.final_model import inference

#maths
import numpy as np
import pandas as pd
import twitter




#Setup
app = Flask(__name__)
CORS(app)

app.config['SERVER_NAME'] = 'api.mota-analytica.io'
 

#Pre-trained Naive-Bayes Classifier
sid = SentimentIntensityAnalyzer()



#Sentiment Analysis 

def basic_sentiment_analysis(review_input):
    sentiment_values = sid.polarity_scores(review_input)
    scoreval = sentiment_values['compound']
    return scoreval



#Get Tweets


#For running sentiment analysis on user input
@app.route("/deepbelief/<user_string>", methods=['GET'])
def analyze_tweets(user_string):

	
    def get_search(user_string):
        t = twitter.Api(consumer_key="",
                consumer_secret="",
                access_token_key="",
                access_token_secret="",
                tweet_mode='extended',
                sleep_on_rate_limit=False,)
  
    #initialize a list to hold all the tweepy Tweets
        all_responses=[]
  

        new_responses = t.GetSearch(term=user_string, count=25, lang='en',result_type='recent')
  
        all_responses.extend(new_responses)
  
  
#         oldest = all_responses[-1].id - 1
  
  
#   #THIS WILL NEED TO BE CHANGED
#         while len(all_responses) < 10:

#             print( "getting tweets before %s" % (oldest))
        
#         #all subsiquent requests use the max_id param to prevent duplicates
#             new_responses = t.GetSearch(term=user_string,count=5,max_id=oldest, lang='en',result_type='recent')
        
#         #save most recent tweets
#             all_responses.extend(new_responses)
        
#         #update the id of the oldest tweet less one
#             oldest = all_responses[-1].id - 1
        
#             print( "...%s tweets downloaded so far" % (len(all_responses)))
	
  
  
        out = []
  
        for tweet in all_responses:
            if tweet.retweeted_status:
                out_list = [tweet.id_str, tweet.created_at, tweet.retweeted_status.full_text]
                out.append(out_list)
            else:
                out_list = [tweet.id_str, tweet.created_at, tweet.full_text]
                out.append(out_list)
  
 
        labels={0: 'tweet_ID',1: 'tweet_date', 2: 'tweet_text'}
        tweet_dict = [{labels[idx]:val for idx,val in enumerate(item)} for item in out]
        return tweet_dict

  
    search = get_search(user_string)
    df = pd.DataFrame.from_dict(search)
    df_test = df.drop(columns='tweet_date')
    df_test.to_csv("/var/www/FlaskApp/FlaskApp/final_model/pybert/dataset/raw/test.csv",index=False)

    result=inference.main()
    df_result = pd.DataFrame(result)
    cols = ["toxic","severe_toxicity","obscene","identity_attack","insult","threat","asian","atheist","bisexual","black","buddhist","christian","female","heterosexual","hindu","homosexual_gay_or_lesbian","intellectual_or_learning_disability","jewish","latino","male","muslim","other_disability","other_gender","other_race_or_ethnicity","other_religion","other_sexual_orientation","physical_disability","psychiatric_or_mental_illness","transgender","white","funny","wow","sad","likes","disagree","sexual_explicit",]
    df_result.columns = cols
    pd.options.display.float_format = '{:.2f}'.format
    df_out = pd.concat([df, df_result], axis=1)
    #df_out.to_csv("/var/www/FlaskApp/FlaskApp/final_model/pybert/dataset/raw/result.csv", index=False)

    #df_out.tweet_ID= pd.to_numeric(df_out.tweet_ID)
   # df_out.tweet_date=pd.to_datetime(df_out.tweet_date)

    #json_out = df_out.to_json()

    #d = df_out.to_dict(orient='records')
    #j = json.dumps(d)

    df_out.set_index('tweet_ID', inplace=True)
    result = {}
    for index, row in df_out.iterrows():
        #result[index] = row.to_json() 
        result[index] = dict(row)
    return jsonify(result)






#Connecting to the database
app.config['MONGO_URI'] = 'mongodb://localhost:27017/yelp'
mongo = PyMongo(app)



@app.route('/')
def get_initial_response():
    """Welcome message for the API."""
    # Message to the user
    message = {
        'API Version': 'v1.0',
        'status': '200',
        'message': 'Welcome to the Mota-Analytica API'
    }
    # Making the message look good
    resp = jsonify(message)
    # Returning the object
    return resp




@app.route("/business/<business_id>", methods=['GET'])
def fetch_reviews(business_id):

    #page = request.args.get('page')
    page = 1
    review = mongo.db.reviews
    
    output = []
    for r in review.find( {"business_id" : business_id} ).skip((page-1)*50).limit(50):
        review_text = r['text']
        adjusted_score = basic_sentiment_analysis(review_text)
        output.append(  {   'score': r['stars'],'adjusted score' : adjusted_score , 'review' : review_text  }   )
    return jsonify({'result' : output})




#For running sentiment analysis on user input
@app.route("/sentiment/<user_string>", methods=['GET'])
def analyze_user_input(user_string):
    user_string = str(user_string)

    user_string = user_string.replace('_', ' ')
    output_text = user_string
    
    user_string = basic_sentiment_analysis(user_string)
    
    return jsonify({output_text : user_string })



#For generating wordclouds
@app.route("/wordcloud/<business_id>")
def mk_cloud(business_id):
   
    business_id = str(business_id)
    fullfilename = "/var/www/FlaskApp/FlaskApp/static/" + business_id + ".png"
    if os.path.isfile(fullfilename) == False:
        review = mongo.db.reviews
        review_text = []
       

        for r in review.find( {"business_id" : business_id} ).limit(50):
            review_text.append(r['text'])
        tokenized_reviews = review_tokenize(review_text)

        cloud_file_name = business_id + ".png"
        cloud_file_path = "static/" + cloud_file_name

        cloud_file_path = cloud_file_path[:-4]

        create_wordcloud(tokenized_reviews, cloud_file_path)
       

        return redirect(url_for('static', filename=cloud_file_name))
    else:
        cloud_file_name = business_id + ".png"
        return redirect(url_for('static', filename=cloud_file_name))



if __name__ == "__main__":
    app.run()
   
