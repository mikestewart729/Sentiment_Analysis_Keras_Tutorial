## Explore the realpython.com Keras tutorial for sentiment analysis using the
## UCI machine learning repository Sentiment Labelled Sentences Data set. See 
## the accompanying readme.md for hyperlinks to tutorial and data sources.

## Import statements
import pandas as pd # Usual convention for importing pandas
from sklearn.model_selection import train_test_split # Split the data
from sklearn.feature_extraction.text import CountVectorizer # bag of words vectors
from sklearn.linear_model import LogisticRegression # logistic regression tool
from keras.models import Sequential # Keras basic sequential model architecture
from keras import layers # Various layers, like Dense and ReLU
from keras.backend import clear_session # To clear the model between runs for tutorial
from keras.preprocessing.text import Tokenizer # Tokenizer similar to nltk
from keras.preprocessing.sequence import pad_sequences # make sentence vectors same size
import matplotlib.pyplot as plt # Convention for importing matplotlib
plt.style.use('ggplot')

clear_session()

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

## Try to improve on the baseline model using a deep network in Keras
df_source = df[df['source'] == 'yelp']

sentences = df_source['sentence'].values # numpy arrays which here are more convenient
y = df_source['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size = 0.25, random_state = 1000
) # Hold out 25% of data and labels for testing, set random state for repeat output in tutorial

tokenizer = Tokenizer(num_words = 5000) # Instantiate a tokenizer to embed the words
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1 # Protected 0 index for OOV token

## Some diagnostic tests and explorations of the embedding
#print(sentences_train[2])
#print(X_train[2])

#for word in ['the', 'all', 'happy', 'sad']:
#    print(f"{word}: {tokenizer.word_index[word]}")

## Pad the sentences so that the vectors all have the same length
max_len = 100

X_train = pad_sequences(X_train, padding = 'post', maxlen = max_len)
X_test = pad_sequences(X_test, padding = 'post', maxlen = max_len)

## Diagnostic to see what our sentences are doing
#print(X_train[0,:])

## Use a Keras embedding layer to "teach" the embedding to the model
embedding_dim = 50
model = Sequential()
model.add(layers.Embedding(
    input_dim = vocab_size, output_dim = embedding_dim, input_length = max_len
))
#model.add(layers.Flatten()) # Naively flatten
model.add(layers.GlobalMaxPool1D()) # Max Pooling to act as a sort of average
model.add(layers.Dense(10, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(
    optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']
)
model.summary()

history = model.fit(
    X_train, y_train, epochs = 20, verbose = False,
    validation_data = (X_test, y_test), batch_size = 10
)
loss, accuracy = model.evaluate(X_train, y_train, verbose = False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))