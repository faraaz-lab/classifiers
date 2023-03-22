from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

import pandas as pd
import numpy as np
import re

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.initializers import Constant
import re

import matplotlib.pyplot as plt
# %matplotlib inline

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix,f1_score

def evaluation_metrics(tn, fp, fn, tp):
    #extracting true_positives, false_positives, true_negatives, false_negatives
#     tn, fp, fn, tp = confusion_matrix(orig_label, predicted_label).ravel()
    print("True Negatives: ",tn)
    print("False Positives: ",fp)
    print("False Negatives: ",fn)
    print("True Positives: ",tp)
    
    #Accuracy
    Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 
    print("Accuracy {:0.2f}%:",format(Accuracy))
    
    #Precision 
    Precision = tp/(tp+fp) 
    print("Precision {:0.2f}".format(Precision))
    #Recall 
    Recall = tp/(tp+fn) 
    print("Recall {:0.2f}".format(Recall))
    
        #F1 Score
    f1 = (2*Precision*Recall)/(Precision + Recall)
    print("F1 Score {:0.2f}".format(f1))
    
    #Specificity 
    Specificity = tn/(tn+fp)
    print("Specificity {:0.2f}".format(Specificity))
    print(str(tn) +'#'+str(fp)+'#'+str(fn)+'#'+str(tp)+'#'+str(Accuracy)+'#'+str(Precision)+'#'+str(Recall)+'#'+str(f1)+'#'+str(Specificity))


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def trim_text(text):
    return(str(text).strip())

dataset=pd.read_pickle("<filename>")

print(dataset.columns)
dataset['NARRATIVE_TEXT']=dataset['nt_new'].apply(trim_text)
dataset['NARRATIVE_TEXT'].replace('', np.nan, inplace=True)
print(dataset.shape)
dataset.dropna(subset=['NARRATIVE_TEXT'], inplace=True)
print(dataset.shape)

df=dataset.copy()
print(df.shape)




df['Phrase']=df.NARRATIVE_TEXT
df['Sentiment']=df[<flagname>]
# pd.set_option('display.max_colwidth', -1)
# df.head(3)

replace_puncts = {'`': "'", '′': "'", '“':'"', '”': '"', '‘': "'"}

strip_chars = [',', '.', '"', ':', ')', '(', '-', '|', ';', "'", '[', ']', '>', '=', '+', '\\', '•',  '~', '@', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

puncts = ['!', '?', '$', '&', '/', '%', '#', '*','£']

def clean_str(x):
    x = str(x)
    
    x = x.lower()
    
    x = re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "url", x)
    
    for k, v in replace_puncts.items():
        x = x.replace(k, f' {v} ')
        
    for punct in strip_chars:
        x = x.replace(punct, ' ') 
    
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
        
    x = x.replace(" '", " ")
    x = x.replace("' ", " ")
        
    return x


df['text'] = df['Phrase'].apply(clean_str)
df['text']=df['text'].apply(trim_text)
df['text'].replace('', np.nan, inplace=True)
print(df.shape)
df.dropna(subset=['text'], inplace=True)
print(df.shape)

df=df.reset_index()

all_train = df.sample(frac=0.70, random_state=0)
all_test = df.drop(all_train.index)



# all_train=df[df.train_valid=="train"]
# all_test=df[df.train_valid=="valid"]
print("all_train",all_train.shape)
print("all_test",all_test.shape)


df_0 = all_train[all_train['Sentiment'] == 0].sample(frac=1)
print(df_0.shape)
df_1 = all_train[all_train['Sentiment'] == 1].sample(frac=1)
print(df_1.shape)
# df_2 = df[df['Sentiment'] == 2].sample(frac=1)
# df_3 = df[df['Sentiment'] == 3].sample(frac=1)
# df_4 = df[df['Sentiment'] == 4].sample(frac=1)

# we want a balanced set for training against - there are 7072 `0` examples
# sample_size = min(len(df_0), len(df_1), len(df_2), len(df_3), len(df_4))
sample_size = min(len(df_0), len(df_1))

data = pd.concat([df_0.head(sample_size), df_1.head(sample_size)]).sample(frac=1)

print(data.shape)

data['l'] = data['Phrase'].apply(lambda x: len(str(x).split(' ')))
print("mean length of sentence: " + str(data.l.mean()))
print("max length of sentence: " + str(data.l.max()))
print("std dev length of sentence: " + str(data.l.std()))

sequence_length = 1029

max_features = 20000 # this is the number of words we care about


tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(data['Phrase'].values)

# this takes our sentences and replaces each word with an integer
X = tokenizer.texts_to_sequences(data['Phrase'].values)

# we then pad the sequences so they're all the same length (sequence_length)
X = pad_sequences(X, sequence_length)

y = pd.get_dummies(data['Sentiment']).values
 
# lets keep a couple of thousand samples back as a test set
print("len(X)",len(X))
print("len(y)",len(y))
# lets keep a couple of thousand samples back as a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)
X_train,  y_train,  = np.array(X),np.array(y)
print("train set size " + str(len(X_train)))
# print("test set size " + str(len(X_test)))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)
# print("train set size " + str(len(X_train)))
# print("test set size " + str(len(X_test)))

import os
embeddings_index = {}
f = open(os.path.join('', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

num_words = min(max_features, len(word_index)) + 1
print(num_words)

embedding_dim = 100

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector

        embedding_matrix[i] = np.random.randn(embedding_dim)
        
vocab_size = 20000  # Max number of different word, i.e. model input dimension
maxlen = 1029  # Max number of words kept at the end of each text

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator

class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """ Sklearn transformer to convert texts to indices list 
    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self
    
    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))
        
sequencer = TextsToSequences(num_words=vocab_size)

class Padder(BaseEstimator, TransformerMixin):
    """ Pad and crop uneven lists to the same length. 
    Only the end of lists longernthan the maxlen attribute are
    kept, and lists shorter than maxlen are left-padded with zeros
    
    Attributes
    ----------
    maxlen: int
        sizes of sequences after padding
    max_index: int
        maximum index known by the Padder, if a higher index is met during 
        transform it is transformed to a 0
    """
    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None
        
    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
        return self
    
    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.maxlen)
        X[X > self.max_index] = 0
        return X

padder = Padder(maxlen)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline

import tensorflow as tf
from tensorflow.keras import backend as K

batch_size = 128
max_features = vocab_size + 1



def create_model(max_features):

    """ Model creation function: returns a compiled Bidirectional LSTM"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=sequence_length,
                        trainable=True))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy',f1])
    return model

# Use Keras Scikit-learn wrapper to instantiate a LSTM with all methods
# required by Scikit-learn for the last step of a Pipeline
sklearn_lstm = KerasClassifier(build_fn=create_model, epochs=10, batch_size=batch_size, 
                               max_features=max_features, verbose=1)

# Build the Scikit-learn pipeline
pipeline = make_pipeline(sequencer,padder,sklearn_lstm)

# pipeline.fit(texts_train, y_train);

pipeline.fit(data['Phrase'].values, pd.get_dummies(data['Sentiment']).values);

X = tokenizer.texts_to_sequences(all_test['Phrase'].values)
# we then pad the sequences so they're all the same length (sequence_length)
X = pad_sequences(X, sequence_length)
y = pd.get_dummies(all_test['Sentiment']).values
X_test,  y_test,  = np.array(X),np.array(y)
print("all_test",all_test.shape)
# y_hat = model.predict(X_test)

from sklearn import metrics
print('Computing predictions on test set...')
y_preds = pipeline.predict(all_test['Phrase'])

y_test=all_test['Sentiment']
y_hat=y_preds
all_test['predicted_label']=y_preds
print('Test accuracy: {:.2f} %'.format(100*metrics.accuracy_score(y_preds, y_test)))
# print("accuracy_score",accuracy_score(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat))))

# conf = confusion_matrix(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat)))
# print(conf)
tn, fp, fn, tp=confusion_matrix(all_test['Sentiment'],all_test['predicted_label']).ravel()
# tn, fp, fn, tp=confusion_matrix(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat)))
evaluation_metrics(tn, fp, fn, tp)
import sklearn
print(sklearn.metrics.classification_report(all_test['Sentiment'],all_test['predicted_label']))

case1=all_test[all_test.new_col=="2015-429456_2517031_AdverseExperienceandEventReportingForm"]
idx=6

class_names=[0,1]

def get_lime_results(case_number):
    case1=all_test[all_test.CASE_NUMBER==case_number]
    idx=case1.index[0]
    print("Narrative Text:",case1.Phrase[idx])
    print("Actual label",case1.Sentiment[idx])
    print("Predicted label",case1.predicted_label[idx])
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=[0,1])
    exp = explainer.explain_instance(case1.Phrase[idx], pipeline.predict_proba, num_features=50)
#     print('Document id: %d' % idx)
    print('Probability(0) =', pipeline.predict_proba([case1.Phrase[idx]])[0,1])
    print('True class: %s' % class_names[case1.Sentiment[idx]])
    print('Predicted class: %s' % class_names[case1.predicted_label[idx]])
#     print("Predicted label",case1.predicted_label[idx])

    print(sorted(exp.as_list(), key = lambda x: x[1],reverse=True))
    import seaborn as sns
    %matplotlib inline
    from collections import OrderedDict
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=[0,1])
    explanation = explainer.explain_instance(case1.Phrase[idx], pipeline.predict_proba, num_features=10)

    weights = OrderedDict(explanation.as_list())
    lime_weights = pd.DataFrame({'words': list(weights.keys()), 'weights': list(weights.values())})

    sns.barplot(x="words", y="weights", data=lime_weights);
    plt.xticks(rotation=45)
    plt.title('Sample {} features weights given by LIME'.format(idx));
    
get_lime_results("2017-065351")
