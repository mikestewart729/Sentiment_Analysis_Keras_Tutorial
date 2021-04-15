## Explore the realpython.com Keras tutorial for sentiment analysis using the
## UCI machine learning repository Sentiment Labelled Sentences Data set. See 
## the accompanying readme.md for hyperlinks to tutorial and data sources.

## Import statements
import pandas as pd # Usual convention for importing pandas
from sklearn.model_selection import train_test_split # Split the data
from sklearn.feature_extraction.text import CountVectorizer # bag of words vectors
from sklearn.linear_model import LogisticRegression # logistic regression tool

## Create a dictionary of all three datafiles we will use
filepath_dict = {
    'yelp': 'data/Sentiment_Analysis/yelp_labelled.txt',
    'amazon': 'data/Sentiment_Analysis/amazon_cells_labelled.txt',
    'imdb': 'data/Sentiment_Analysis/imdb_labelled.txt'
}

## Read in the data using pandas and concatenate into one dataframe
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names = ['sentence', 'label'], sep = '\t')
    df['source'] = source # Add a column with the data source: imdb, etc.
    df_list.append(df)

df = pd.concat(df_list)
#print(df.iloc[0]) # Test that the inputs are working

## Define the baseline model using a bag of words approach and 
## logistic regression. 
## Try it out on just the yelp sentences first. Then modify the 
## code to run the basic classifier on each text type.
for source in df['source'].unique():
    df_source = df[df['source'] == source]

    sentences = df_source['sentence'].values # numpy arrays which here are more convenient
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size = 0.25, random_state = 1000
    ) # Hold out 25% of data and labels for testing, set random state for repeat output in tutorial

    vectorizer = CountVectorizer() # Instantiate a count vectorizer
    vectorizer.fit(sentences_train) # Train it on the training sentences

    X_train = vectorizer.transform(sentences_train) # Transform training sentences into features
    X_test = vectorizer.transform(sentences_test) # Transform test sentences into features

    classifier = LogisticRegression(solver = 'lbfgs') # Instantiate our base logistic regression
    classifier.fit(X_train, y_train) # Train the classifier
    score = classifier.score(X_test, y_test) # Score the classifier on the test set

    print(f"Accuracy for {source} data: {score:.4f}")