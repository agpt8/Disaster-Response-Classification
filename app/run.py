import json

import joblib
import nltk
import numpy as np
import pandas as pd
import plotly
from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar, Pie
from scipy.stats.mstats import gmean
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import fbeta_score
from sqlalchemy import create_engine

# nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

app = Flask(__name__)


def tokenize(text):
    """
    Tokenize function to process text data including lemmatize, normalize
    case, and remove leading/trailing white space

    Args:
        text (str): list of text messages (english)

    Returns:
        clean_tokens: tokenized text, clean and ready to feed ML modeling
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    @staticmethod
    def isverb(text):
        """
        Check if the starting word is a verb
        Args:
            text (str): text messages to be checked

        Returns:
            boolean (bool)
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # def fit(self, x, y=None):
    #     """
    #     Fit the model
    #     """
    #     model.fit(x,y)
    #     return self

    def transform(self, x):
        """
          Transform incoming dataframe to check for starting verb
           Args:
              x:
           Returns:
              dataframe with tagged verbs (pd.Dataframe)
          """
        x_tagged = pd.Series(x).apply(self.isverb)
        return pd.DataFrame(x_tagged)


def multioutput_fscore(y_true, y_pred, beta=1):
    """
    This is a performance metric reference.
    It is a sort of geometric mean of the fbeta_score, computed on each label.
    It is compatible with multi-label and multi-class problems.
    It features some peculiarities (geometric mean, 100% removal...) to exclude
    trivial solutions and deliberately under-estimate a standard fbeta_score average.
    The aim is avoiding issues when dealing with multi-class/multi-label imbalanced cases.
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore, beta=1)

    Args:
        y_true: labels
        y_pred: predictions
        beta: beta value of fscore metric

    Returns:
        customized fscore
    """
    score_list = []
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    for column in range(0, y_true.shape[1]):
        score = fbeta_score(y_true[:, column], y_pred[:, column], beta,
                            average='weighted')
        score_list.append(score)
    f1score_numpy = np.asarray(score_list)
    f1score_numpy = f1score_numpy[f1score_numpy < 1]
    f1score = gmean(f1score_numpy)
    return f1score


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('message_categories', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Index page of the app

    Returns:
        Render template for the home page
    """
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = df.iloc[:, 4:].columns
    category_counts = (df.iloc[:, 4:] != 0).sum().values
    category_percentage = category_counts / category_counts.sum()

    # create visuals
    graphs = [
        # Graph 1: Genre Graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # Graph 2: Category Graph
        {
            'data': [
                Pie(
                    labels=category_names,
                    values=category_percentage
                )
            ],
            'layout': {'title': 'Percentage of Message Categories', 'height': 550},
            'textinfo': "label+percent",
            'textposition': "outside",
        },

        # Graph 3: Category Graph
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Message count of each Categories"
                },
                'xaxis': {
                    'title': "Messages Category",
                    'tickangle': 30
                }
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
    """
    Function to show the results page when a user has put a message in the quert box

    Returns:
        Render template for the results page
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template('go.html', query=query, classification_result=classification_results)


def main():
    """
    Main function to run the app
    """
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
