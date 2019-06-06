Hi Professor, AI's

Greetings of the day !!!

Box link for the project:
https://iu.box.com/s/hx0mc7yvvkhxez25qjpvvnt5i3zwxaqi

First step is to install Flask and any prompted libraries, if not installed. Please follow the below commands taken from the below link.		
                      a. On command prompt window, type the below command:
				pip install flask 
                      or follow the instructions in the following link
                      b. http://flask.pocoo.org/docs/1.0/installation/

2. Download the ‘flask-by-example' folder from the "ML in CL Project" folder uploaded in Box.

3. Go inside the "api" folder and run the python script “form.py”. :  py form.py

4. Open the below link in the browser :  http://127.0.0.1:5000/
		
5. Now you can search by the song title or artist name. Alternatively, you can click on the "Top Lyrics", "New Lyrics" icon located at the end of the web page.

6. A new page will be opened displaying the top 10 songs along with their Mood and Genre Classification as per the search criteria. 

7. Users can give their feedback for the Mood Predictions by clicking on the right or wrong icons at the last column and then submit their feedback.

8. Now to retrain the model (dynamic learning ) with the feedback, run the python script "bestModel_mood.py"

Due to memory constraints we have performed our experimentation on google colab notebooks. The two colab note books are:
https://drive.google.com/open?id=1xDc4icjEfYMbRgSOY8NF_wP4qnT4_t6h
https://drive.google.com/open?id=1rWHpaIHaRc2Ku5Ky2v_WesXNrmXoD9uE

The data files required to run the colab notebooks are uploaded in the “Datasets for Colab Notebooks” folder , they are :
train_lyrics_1000.csv
valid_lyrics_200.csv
train_cleaned_lyrics.tsv
glove.6B.100d.txt

After the experimentation we deployed the best model (Linear SVC ) for both mood and genre classification we have implemented them in .py files in the “flask-by-example/api” folder.
bestModel_mood.py
bestModel_genre.py

Thanks & Regards,
Ashlesha Shinde
Chhavi Sharma
Prahasan Gadugu
Supriya Ayalur Balasubramanian