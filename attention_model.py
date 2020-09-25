# Import all the required packages 

import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unicodedata
import re
import os
import io
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Embedding,LSTM, Dense
import pickle
# Reference - https://medium.com/generalist-dev/background-colour-detection-using-opencv-and-python-22ed8655b243
import sys
import cv2
from collections import Counter

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  w = w.strip()
  w = '<start> ' + w + ' <end>'
  return w

class Encoder_att(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):

        #Initialize Embedding layer
        #Intialize Encoder LSTM layer
        super(Encoder_att, self).__init__()
        self.inp_vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.input_length = input_length
        self.lstm_size= lstm_size
        self.embedding = tf.keras.layers.Embedding(input_dim=self.inp_vocab_size, output_dim=self.embedding_size, name="embedding_layer_encoder")
        self.lstm = tf.keras.layers.LSTM(self.lstm_size, return_state=True, return_sequences=True, name="Encoder_LSTM", recurrent_initializer='glorot_uniform')

    def call(self,input_sequence,states):
      '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- All encoder_outputs, last time steps hidden and cell state
      '''
      input_embedd = self.embedding(input_sequence)
      lstm_output, lstm_state_h, lstm_state_c = self.lstm(input_embedd,initial_state = states)
      return lstm_output, lstm_state_h, lstm_state_c
    
    def initialize_states(self,batch_size):
      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state shape is [32,lstm_units], cell state shape is [32,lstm_units]
      '''
      return tf.zeros((batch_size, self.lstm_size)) , tf.zeros((batch_size, self.lstm_size))

class Attention(tf.keras.layers.Layer):
  '''
    This Class calculates score using Bahdanu attention mechanism.
  '''
  def __init__(self, att_units):
    
      super(Attention, self).__init__()
      # Intialize variables needed for Concat score function here
      self.W1 = tf.keras.layers.Dense(att_units)
      self.W2 = tf.keras.layers.Dense(att_units)
      self.V = tf.keras.layers.Dense(1)  
  
  def call(self,decoder_hidden_state,encoder_output):

    '''
        This function takes the encoder output and decoder hidden state as input and calculate the 
        score using Attention concat scoring function. This scores will be used to attention weights using a softmax layer.
        Then This attention weights will be used to generate the contenxt vector by multiplying with encoder output.
    '''
    # Implement concat score function here
    # Extend the decoder hidden state dimension
    decoder_hidden_state = tf.expand_dims(decoder_hidden_state, 1)
    # apply the decoder hidden state and encoder output to dense layer and then sum both of them
    score = self.V(tf.nn.tanh(self.W1(decoder_hidden_state) + self.W2(encoder_output)))

    # Get the attention weights after applying softmax on the scores 
    attention_weights = tf.nn.softmax(score, axis=1)
    # Generate the context vector by multiplying encoder output and attention weights
    context_vector = attention_weights * encoder_output
    context_vector = tf.reduce_sum(context_vector, axis=1)
    
    return context_vector,attention_weights

class One_Step_Decoder(tf.keras.Model):
  '''
    This class takes decoder input, encoder output and hiddena and cell states and return decoder output along with attention weights
  '''
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,att_units):
  # Initialize decoder embedding layer, LSTM and any other objects needed
    super(One_Step_Decoder, self).__init__()
    self.tar_vocab_size = tar_vocab_size
    self.embedding_dim = embedding_dim
    self.input_length = input_length
    self.dec_units = dec_units
    self.att_units = att_units
    self.embedding = tf.keras.layers.Embedding(input_dim=self.tar_vocab_size, output_dim=self.embedding_dim, name="Decoder_Embedding_layer")
    self.lstm = tf.keras.layers.LSTM(self.dec_units, return_state=True,return_sequences=True, name="Decoder_LSTM", recurrent_initializer='glorot_uniform')
    self.dense = tf.keras.layers.Dense(self.tar_vocab_size, name="DenseOut")

    self.attention = Attention(self.att_units)


  def call(self,input_to_decoder, encoder_output, state_h,state_c):
    '''
        One step decoder mechanisim step by step:
      A. Pass the input_to_decoder to the embedding layer and then get the output(1,1,embedding_dim)
      B. Using the encoder_output and decoder hidden state, compute the context vector.
      C. Concat the context vector with the step A output
      D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
      E. Pass the decoder output to dense layer(vocab size) and store the result into output.
      F. Return the states from step D, output from Step E, attention weights from Step -B
    '''
    input_embedd = self.embedding(input_to_decoder)
    context_vector, attention_weights = self.attention(state_h , encoder_output )
    concat = tf.concat([tf.expand_dims(context_vector, 1), input_embedd], axis=-1)
    decoder_output, dec_state_h, dec_state_c = self.lstm(concat)   #, initial_state = (state_h,state_c))

    decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
    output = self.dense(decoder_output)

    return output,dec_state_h, dec_state_c, attention_weights, context_vector


def predict_att(input_sentence, encoder, one_Step_Decoder,inp_lang, targ_lang, max_length_inp, max_length_targ):

  '''
    A. Given a sentence first preprocess it 
    B. Convert the sentence to tokens, add <'start'> and <'end'> tokens
    C. Initialize encoder state 
    D. Pass the input data to encoder class along with encoder initial state
    E. Pass token of <'start'> to one_step_decoder as initial stage
    F. Get the predicted next token and pass it in next loop. Run this untill we get a end token of <'end'>
    G. Convert all the output tokens to output text
    G. Return the predicted text 
   '''
  # Create weights matrics
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  # preprocess the input text (if already preprocess then comment this code)
  input_sentence = preprocess_sentence(input_sentence)

  # convert all input text to respected tokens and padd the rest
  #inputs = [inp_lang.word_index[i] for i in input_sentence.split(' ')]
  inputs = []
  for i in input_sentence.split(' '):
      try :
          inputs.append(inp_lang.word_index[i])
      except :
          continue
      
  input_sentence_pad = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')

  input_sentence_pad = tf.convert_to_tensor(input_sentence_pad)

  # Get the encoder 
  ini_state = encoder.initialize_states(1)

  encoder_output,state_h,state_c = encoder(input_sentence_pad, ini_state)
  dec_state_h = state_h
  dec_state_c = state_c

  result = ''
  # predict next word till we get <end>
  dec_input = tf.expand_dims( [targ_lang.word_index['<start>']] , 0)

  for t in range(max_length_targ):
      # passing enc_output to the decoder
      predictions,dec_state_h,_,attention_weights, context_vector = one_Step_Decoder(dec_input, encoder_output, dec_state_h,dec_state_c )

      attention_weights = tf.reshape(attention_weights, (-1, ))
      attention_plot[t] = attention_weights.numpy()

      predicted_id = tf.argmax(predictions[0]).numpy()
      result += targ_lang.index_word[predicted_id] + ' '

      if targ_lang.index_word[predicted_id] == '<end>':
        return result, input_sentence, attention_plot

      # the predicted ID is fed back into the model
      dec_input = tf.expand_dims([predicted_id], 0)

  return result, input_sentence, attention_plot
  
# Define function to translate the input text
def translate_sent(ita_sent, encoder,onestepdecoder,inp_lang, targ_lang, max_length_inp, max_length_targ):
    
    # If multiple sentence sent as input, split it and process separately
    ita_sent = re.sub(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s","\n",ita_sent)
    final_res = ''
    for inp in ita_sent.split("\n"):
        eng,_,_ = predict_att(inp , encoder, onestepdecoder,inp_lang, targ_lang, max_length_inp, max_length_targ)
        eng = eng.replace('<start>', '').strip()
        eng = eng.replace('<end>', '').strip()
        eng = eng.split()
          
        result = ''
        for i in eng:
          result += i + ' '
        result = result.replace(" .",".")
        result = result.replace(" ?","?")
        result = result.replace(" ,",",")
        result = result.replace(" !","!")
        result = result.replace(" t ","'t ")
        result = result.replace(" s ","'s ")
        result = result.replace(" ll ","'ll ")
        result = result.replace(" re ","'re ")
        result = result[0].upper()+result[1:]
        result = result.strip()
        final_res += ' '+result
    final_res = final_res.strip()
    return final_res

# Get the background color of a text
class GetBGColor():
    def __init__(self, img):
        self.img = img
        self.manual_count = {}
        self.w, self.h, self.channels = self.img.shape
        self.total_pixels = self.w*self.h

    # get the count of BGR for each pixel
    def count(self):
        for y in range(0, self.h):
            for x in range(0, self.w):
                BGR = (self.img[x, y, 0], self.img[x, y, 1], self.img[x, y, 2])
                if BGR in self.manual_count:
                    self.manual_count[BGR] += 1
                else:
                    self.manual_count[BGR] = 1
    
    # Get the average color if most color probability is less than 0.5
    def average_colour(self):
        r = 0
        g = 0
        b = 0
        sample = 20
        for top in range(0, sample):
            b += self.number_counter[top][0][0]
            g += self.number_counter[top][0][1]
            r += self.number_counter[top][0][2]
        return (b/sample, g/sample, r/sample)

    # Find the most probable color
    def detect(self):
        self.count()
        self.number_counter = Counter(self.manual_count).most_common(20)
        self.percentage_of_first = (float(self.number_counter[0][1])/self.total_pixels)

        if self.percentage_of_first > 0.5:
            return self.number_counter[0][0]
        else:
            return self.average_colour()


