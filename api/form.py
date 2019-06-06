# https://medium.com/@anuragdhar1992/music-mood-app-2e70886ba550 citation

from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from bs4 import BeautifulSoup
import requests
import re
#import pickle
import pandas as pd
#import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#from flask_json import FlaskJSON, JsonError, json_response, as_json
from sklearn.externals import joblib
#import csv
 
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
load_model_genre = joblib.load("pipeline_genre.pkl")
load_model_mood = joblib.load("pipeline_mood.pkl")

class ReusableForm(Form):
    artist = TextField('Artist: ', validators=[validators.required()])
    #songname = TextField('Song Title: ', validators=[validators.required()])

def preprocess(given_review):
    review = given_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))



def predict(lyrics):
#    cleaned_test_lyrics=preprocess(lyrics)
#    cv=pickle.load(open("naivebCV.pkl","rb"))
#    vectorized_test_lyrics=cv.transform([lyrics])
#    loaded_model = pickle.load(open("naivebmodel.pkl","rb"))
    result = load_model_mood.predict([lyrics])
    return result[0]    

def predict_pipe(lyrics):
#    cleaned_test_lyrics=preprocess(lyrics)
    
    result_genre = load_model_genre.predict([lyrics]) 
    return result_genre[0]

def clean_lyrics(lyrics_list):
    cleaned_lyrics=[]
    for lyrics in lyrics_list:
        cleaned_lyrics.append(preprocess(lyrics))
    return cleaned_lyrics
    

def get_mood(lyrics_list):
    mood=[]
    for lyrics in lyrics_list:
        mood.append(predict(lyrics))
    return mood

def get_genre(lyrics_list):
    genre=[]
    for lyrics in lyrics_list:
        genre.append(predict_pipe(lyrics))
    return genre

def web_scrape(url,search_type):
        headers = {'User-Agent':'Mozilla/5.0'}
        req=requests.get(url,headers=headers)
        parser = BeautifulSoup(req.text, 'html.parser')
        links_for_lyrics = parser.find_all('h2', {'class': 'media-card-title'})
        print(len(links_for_lyrics))
        artist_name = parser.find_all('h3', {'class': 'media-card-subtitle'})
        print(len(artist_name))
        url_for_icon = parser.find_all('div', {'class': 'media-card-picture'})
        print(len(url_for_icon))
        href_list=[] 
        icon_list=[]
        artist_ref_list=[]
        song_name_list=[]
        artist_name_list=[]
        lyrics_list=[]
        song_nos=10
        x=0
        if search_type=="search":
            x=1
        for count,links in enumerate(links_for_lyrics[1:]):
            if count < int(song_nos):
                anchor = links.find('a')
                song_name_list.append(anchor.text)
                href = anchor['href']
                href_list.append(href)
        
        for count1,icon in enumerate(url_for_icon[x:]):
            if count1 < int(song_nos):
                img=icon.find('img')
                icon_url = img['srcset']
                icon_list.append(icon_url[0:68])
#        flash(href_list)
        for count2,artist in enumerate(artist_name):
            if count2 < int(song_nos):
                art = artist.find('a')
                artist_name_list.append(art.text)
                artist_ref = art['href']
                artist_ref_list.append(url+artist_ref)
        lyrics_url="https://www.musixmatch.com"
        for href in href_list:
            lyrics=''
            req=requests.get(lyrics_url+href,headers=headers)
            html = BeautifulSoup(req.text, 'html.parser')
            text = html.find_all('span',{'class': 'lyrics__content__ok'})
            for line in text:
                lyrics+=line.get_text()
            lyrics_list.append(lyrics)
        return(href_list,icon_list,artist_ref_list,song_name_list,artist_name_list,lyrics_list)
    

@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
#    print (form.errors)
    if request.method == 'POST':
#        songname=request.form.get('songname', None)#['songname']
        artist=request.form.get('artist',None)#['artist']
#        print (songname)
        
    if form.validate():
        url = "https://www.musixmatch.com/search/"
        
        singer=artist.split()
#        print(artist)
#        flash(artist)
        for i in range(0,len(singer)):
            if i==0:
                url+=singer[i].lower()
            else:
                url+="%20"+singer[i].lower()
        href_list=[] 
        icon_list=[]
        artist_ref_list=[]
        song_name_list=[]
        artist_name_list=[]
        lyrics_list=[]
        
        href_list,icon_list,artist_ref_list,song_name_list,artist_name_list,lyrics_list = web_scrape(url,"search")
        
        column_list = ["id",'title', 'artist', 'lyrics_url', 'lyrics', 'icon_url', 'mood','genre']
        songs_df = pd.DataFrame(columns=column_list)
        songs_df['id'] = list(range(len(song_name_list)))
        songs_df['title'] = song_name_list
        songs_df['artist'] = artist_name_list
        songs_df['lyrics_Url'] = href_list
        songs_df['icon_url'] = icon_list
        songs_df['lyrics'] = lyrics_list
        cleaned_lyrics = clean_lyrics(lyrics_list)
        mood = get_mood(cleaned_lyrics)
        genre = get_genre(cleaned_lyrics)
        songs_df['mood'] = mood
        songs_df['genre'] = genre
        data = songs_df.to_dict(orient="records")
            #print(data)
        headers = songs_df.columns
#        print(request.path)
        return render_template("table.html", data=data, headers=headers)
    
    return render_template('index.html', form=form)

@app.route("/topLyrics/<list_type>", methods=['GET', 'POST'])
def topLyrics(list_type):
#    flash("topLyrics")
#    if request.method =='POST':
        if list_type=="toplyrics":
            url = "https://www.musixmatch.com/explore"
        elif list_type =="newlyrics":
            url = "https://www.musixmatch.com/explore/"+list_type
            #    print (url)
        href_list,icon_list,artist_ref_list,song_name_list,artist_name_list,lyrics_list = web_scrape(url,"list")
        column_list = ['id','title', 'artist', 'lyrics_url', 'lyrics', 'icon_url', 'mood']
        songs_df = pd.DataFrame(columns=column_list)
        songs_df['id'] = list(range(len(song_name_list)))
        songs_df['title'] = song_name_list
        songs_df['artist'] = artist_name_list
        songs_df['lyrics_Url'] = href_list
        songs_df['icon_url'] = icon_list
        songs_df['lyrics'] = lyrics_list
        
        cleaned_lyrics = clean_lyrics(lyrics_list)
        mood = get_mood(cleaned_lyrics)
        genre = get_genre(cleaned_lyrics)
        songs_df['mood'] = mood
        songs_df['genre'] = genre
        
#        mood=get_mood(lyrics_list)
#        songs_df['mood']=mood
        data = songs_df.to_dict(orient="records")
            #print(data)
        headers = songs_df.columns
        return render_template("table.html", data=data, headers=headers)
#    return render_template('index.html', form=form)

#@app.route("/feedback/<url>", methods=['GET', 'POST'])
#def feedback(url):
##    print(url)
#    headers = {'User-Agent':'Mozilla/5.0'}
#    if url=="home":
#        url="http://127.0.0.1:5000/"
#    elif url=="newlyrics":
#        url="http://127.0.0.1:5000/topLyrics/newlyrics"
#    else:
#        url="http://127.0.0.1:5000/topLyrics/toplyrics"
#    
#    req=requests.get(url,headers=headers)
#    parser = BeautifulSoup(req.text, 'html.parser')
#    table = parser.find("table")
#    output_rows = []
#    for table_row in table.findAll('tr')[1:]:
#        columns = table_row.findAll('td')
#        output_row = []
#        for column in columns:
#            output_row.append(column.text)
##        hidden=table_row.find("input")#, type="hidden")
##        output_row.append(hidden["value"])
#        output_rows.append(output_row)
##    print(output_rows)s
#    df = pd.DataFrame(output_rows)
#    df.to_csv('out.csv',index = False)
#    return render_template("hiii.html")
#df = pd.DataFrame()
@app.route("/feedback", methods=['POST'])
def feedback():

    if request.method=="POST":
#        html=request.get_json()
#        row=[]
#        column_list = ['id','Cover','Title','Artist','Mood','Prediction','Feedback Yes','Feedback No','Answer']
#        df = pd.DataFrame(column=column_list)
#        row.append(request.form["javascript_data"])
#        
#        df.append(row)
        html=request.form["javascript_data"]
#        flash(html)
        print(html)
#        changed=pd.DataFrame()
#        changed=pd.read_html(html)
#        print (changed)
#        df = pd.DataFrame(changed)
#        df.to_csv('out_puttab.csv',index = False)
#        df = pd.DataFrame(html)
#        df.to_csv('out_table.csv',index = False)
        df_table=[]
        for row in html.split('],['):
            df_row=[]
            for col in row.split('","'):
                col=col.replace('"','')
                col=col.replace('\n','')
#                col = re.sub('[^a-zA-Z]', ' ',col)
                if col=='' or col=='[[' or col==']]':
                   pass
                else:
                    df_row.append(col)
            df_table.append(df_row)
        #return (df_table)
#       df.append(df_table)
        df = pd.DataFrame(df_table)
        print(df)
        with open('feedback.csv','a') as file:
            df.to_csv(file,index = False)
        
    return render_template("hiii.html")
    #return(df_table)
     
if __name__ == "__main__":
  app.run()
  
#lyr = feedback()


loaded_model = joblib.load("pipeline_genre.pkl")
k=loaded_model.predict(["jkasfjndfd"])