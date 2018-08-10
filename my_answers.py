import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    t = np.asarray(series)
    
    for i in range(0,t.size-window_size,1):
        X.append(t[i:i+window_size])
        y.append(t[i+window_size])
        
    

    # reshape each 
    X = np.asarray(X)
    y = np.asarray(y)
    y = y.reshape((t.size-window_size,1))
    
    return X,y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    #pass
    model = Sequential()
    model.add(LSTM(5,input_shape=(window_size,1)))
    #mdel.add(LSTM(5))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    chars_and_space = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', ' ']
    s1 = set(chars_and_space)
    s2 = set(punctuation)
    text = ''.join([c for c in text if c in list(s1.union(s2))])
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    i = 0
    import math
    n = math.ceil((len(text)-window_size)/step_size)
    while n > 0:
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
        i = i+step_size
        n = n-1

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    #pass
    model = Sequential()
    model.add(LSTM(200,input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model