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
from keras.wrappers.scikit_learn import KerasClassifier # Used to run grid search
from sklearn.model_selection import RandomizedSearchCV # Actual parameter search
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

## Set up to do a grid search in the CNN models
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, max_len):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length = max_len))
    model.add(layers.Conv1D(num_filters, kernel_size, activation = 'relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(
        optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']
    )
    return model

## Main settings
epochs = 20
embedding_dim = 50
max_len = 100
output_file = 'data/output.txt'

## Run grid search for each data source
for source, frame in df.groupby('source'):
    print('Running grid search for data set :', source)
    sentences = df['sentence'].values
    y = df['label'].values

    # Train-test split
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size = 0.25, random_state = 1000
    )

    # Tokenize words
    tokenizer = Tokenizer(num_words = 5000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    # Get vocab size and remember to add 1 for OOV token
    vocab_size = len(tokenizer.word_index) + 1

    # Pad sentences with zeros
    X_train = pad_sequences(X_train, padding = 'post', maxlen = max_len)
    X_test = pad_sequences(X_test, padding = 'post', maxlen = max_len)

    # Parameter grid for grid search
    param_grid = dict(
        num_filters = [32, 64, 128],
        kernel_size = [3, 5, 7],
        vocab_size = [vocab_size],
        embedding_dim = [embedding_dim],
        max_len = [max_len]
    )

    model = KerasClassifier(
        build_fn = create_model, epochs = epochs, batch_size = 10, verbose = False
    )

    grid = RandomizedSearchCV(
        estimator = model, param_distributions = param_grid, cv = 4, verbose = 1, n_iter = 5
    )

    grid_result = grid.fit(X_train, y_train)

    # Evaluate the testing set
    test_accuracy = grid.score(X_test, y_test)

    # Save and evaluate the results
    prompt = input(f"Finished {source}; write to a file and proceed? [y/n]")
    if prompt.lower() not in {'y', 'true', 'yes'}:
        break
    with open(output_file, 'a') as f:
        s = ("Running {} data set\nBest accuracy: {:.4f}\n{}\nTest accuracy: {:.4f}\n\n")
        output_string = s.format(
            source,
            grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy
        )
        print(output_string)
        f.write(output_string)