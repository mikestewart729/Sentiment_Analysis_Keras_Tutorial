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
import matplotlib.pyplot as plt # Convention for importing matplotlib
plt.style.use('ggplot')

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

vectorizer = CountVectorizer() # Instantiate a count vectorizer
vectorizer.fit(sentences_train) # Train it on the training sentences

X_train = vectorizer.transform(sentences_train) # Transform training sentences into features
X_test = vectorizer.transform(sentences_test) # Transform test sentences into features

input_dim = X_train.shape[1] # Number of features

clear_session()

## Build the model
model = Sequential()
model.add(layers.Dense(10, input_dim = input_dim, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

## Compile the model
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

model.summary()

## Train the model
history = model.fit(
    X_train, y_train, epochs = 100, verbose = False, 
    validation_data = (X_test, y_test), batch_size = 10
)

## Evaluate the model accuracy
loss, accuracy = model.evaluate(X_train, y_train, verbose = False)
print(f"Training accuracy: {accuracy:.4f}")
loss, accuracy = model.evaluate(X_test, y_test, verbose = False)
print(f"Test accuracy: {accuracy:.4f}")

## Helper function to visualize the training
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label = 'Training Acc.')
    plt.plot(x, val_acc, 'r', label = 'Validation Acc.')
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label = 'Training Loss')
    plt.plot(x, val_loss, 'r', label = 'Validation Loss')
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

#plot_history(history)