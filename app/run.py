#%%pycodestyle
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Cleaned_Messages', engine)
df.to_csv("test.csv")

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # calculate all messages count for pie chart 
    genre_counts = df.groupby('genre').count()['message']
    gen_percentage = round(100*genre_counts/genre_counts.sum(), 3)
    genre_names = list(genre_counts.index)
    # calculate essages count for bar chart that includes/not include 
    # the word 'medical_help'
    genre_bar_1_df = df.groupby(['genre', 'medical_help']). \
        count()['message'].to_frame().reset_index()
    genre_bar_1_df["gen_med_help"] = [str(genre_bar_1_df['genre'][c])\
                        + "_" + str(genre_bar_1_df['medical_help'][c]) \
                                    for c in range(len(genre_bar_1_df))]

    # calculate messages count for pie chart that includes 
    # the word 'medical_help'
    genre_bar_1_df_1 = genre_bar_1_df[genre_bar_1_df.medical_help != 0]
    genre_counts_2 = genre_bar_1_df_1['message']
    gen_percentage_2 = round(100*genre_counts_2/genre_counts_2.sum(), 3)
    genre_names_2 = list(genre_counts.index)

    # calculate essages count for bar chart that includes/not include 
    # the word 'missing_people'
    genre_bar_2_df = df.groupby(['genre', 'missing_people']). \
                count()['message'].to_frame().reset_index()
    genre_bar_2_df["gen_miss_ppl"] = [str(genre_bar_2_df['genre'][c]) + \
                        "_"+str(genre_bar_2_df['missing_people'][c])  \
                        for c in range(len(genre_bar_2_df))]

    # calculate messages count for pie chart that includes 
    # the word 'missing_people'
    genre_bar_2_df_1 = genre_bar_2_df[genre_bar_2_df.missing_people != 0]
    genre_counts_3 = genre_bar_2_df_1['message']
    gen_percentage_3 = round(100*genre_counts_3/genre_counts_3.sum(), 3)
    genre_names_3 = list(genre_counts.index)

    # create visuals
    graphs = [
        {
            "data": [
              {
                "type": "pie",
                "hole": 0.4,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": gen_percentage,
                  "y": genre_names
                },
                "marker": {
                  "colors": [
                    "#17becf",
                    "#e377c2",
                    "#bcbd22"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names,
                "values": genre_counts
              }
            ],
            "layout": {
              "title": "The Overall Message Precentage by Genre"
            }
        },
    # Add "medical_help" include in the messages
        {
            'data': [
                Bar(
                    x=genre_bar_1_df.gen_med_help.values,
                    y=genre_bar_1_df.message.values
                )
            ],
            'layout': {
                'title': 'Count of Messages based on \'medical_help\' ' + 
                            'word included (1 = Yes) for each \'Genre\' ',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "\'medical_help\' word included in the message"
                }
        }
        },
        {
            "data": [
              {
                "type": "pie",
                "hole": 0.4,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": gen_percentage_2,
                  "y": genre_names_2
                },
                "marker": {
                  "colors": [
                    "#17becf",
                    "#e377c2",
                    "#bcbd22"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names_2,
                "values": genre_counts_2
              }
            ],
            "layout": {
              "title":" The precentage of the \'medical_help\' word included in the messages"
            }
        },
        
        # Add "missing_people" include in the messages
        {
            'data': [
                Bar(
                    x=genre_bar_2_df.gen_miss_ppl.values,
                    y=genre_bar_2_df.message.values
                )
            ],
            'layout': {
                'title': 'Count of Messages based on \'missing_people\' word included (1 = Yes) for each \'Genre\' ',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "\'missing_people\' word included in the message"
                }
            
        }
        },
        {
            "data": [
              {
                "type": "pie",
                "hole": 0.4,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": gen_percentage_3,
                  "y": genre_names_3
                },
                "marker": {
                  "colors": [
                    "#17becf",
                    "#e377c2",
                    "#bcbd22"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names_3,
                "values": genre_counts_3
              }
            ],
            "layout": {
              "title": " The precentage of the \'missing_people\' word included in the messages"
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()