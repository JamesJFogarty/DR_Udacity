import json
import plotly
import pandas as pd
import numpy as np

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
engine = create_engine('sqlite:///../data/database_file.db')
df = pd.read_sql_table('messages_disaster', engine)


# load model
model = joblib.load("../data/model.pkl")


# Message list
message_list = ['related', 'request', 'offer', 'aid_related',
          'medical_help', 'medical_products', 'search_and_rescue',
          'security', 'military', 'child_alone', 'water', 'food',
          'shelter', 'clothing', 'money', 'missing_people', 'refugees',
          'death', 'other_aid', 'infrastructure_related', 'transport',
          'buildings', 'electricity', 'tools', 'hospitals', 'shops',
          'aid_centers', 'other_infrastructure', 'weather_related',
          'floods', 'storm', 'fire', 'earthquake', 'cold',
          'other_weather', 'direct_report']

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_group = df.groupby('genre', as_index=False).agg({'message':'count'})
    genre_group['proportion'] = np.round(genre_group['message'] / (genre_group['message'].sum() * 1.0),1)
    genre_prop = genre_group['proportion'].unique().tolist()
    genre_names = genre_group['genre'].unique().tolist()
    
    message_sum_list = [df[m].sum() for m in message_list]
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_prop
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=message_list,
                    y=message_sum_list
                )
            ],

            'layout': {
                'title': 'Distribution of Messages recived (based on category)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
      
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()