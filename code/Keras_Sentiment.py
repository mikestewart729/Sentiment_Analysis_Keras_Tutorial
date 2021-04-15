## Explore the realpython.com Keras tutorial for sentiment analysis using the
## UCI machine learning repository Sentiment Labelled Sentences Data set. See 
## the accompanying readme.md for hyperlinks to tutorial and data sources.

## Import statements
import pandas as pd # Usual convention for importing pandas

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

## Define the baseline model using a bag of words approach