#%%pycodestyle
# import libraries
import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import re
import pickle
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Load and given dataset from the database
    input:
         database name
    outputs:
        X: messages info
        y: features to predict
        category names: category names from y
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Cleaned_Messages', engine)
    X = df['message']
    y =  df[df.columns[4:]]
    
    category_names = list(np.array(y.columns))

    return X, y, category_names


def tokenize(text):
    """  The funtion resturn the 'message' text words normalize and tokanize 
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    wrd_tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for tok in wrd_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(clean_tok)

    return cleaned_tokens


def build_model():
    """
    This funtion generates the pipeline for the model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'vect__min_df': [1, 7],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [2, 20],
                'clf__estimator__min_samples_split': [4, 10]}

    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates trained model
    inputs
        model: Trained model
        X_test: Test dataset features
        y_test: Test dataset values
        category_names: values of the y 
    output:
        scores
    """
    y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, y_pred, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    """
    This funtion save trained model to a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()