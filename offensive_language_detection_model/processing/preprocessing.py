#This module contains functions that will be used to preprocess data before passing to the model

from typing import List
import pandas as pd
import contractions
import emoji
import string
from word2number import w2n
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.corpus import stopwords
import re
import tensorflow_hub as hub
import tensorflow_text as text
from offensive_language_detection_model.config.core import config

import nltk

#not sure if these download statements should be here or in the environment
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

tfhub_preprocess = config.model_config.bert_preprocessor_url
bert_tokenizer = hub.KerasLayer(tfhub_preprocess)

#function to normalize the text

def lowercase(text):

    """ Function_name: normalizer
        input_params:
            text: string
        output_params: text in lowercase
        """
    return text.lower()

#function to remove emojis

def remove_emoji(text):

    """ Function_name: emoji_remover
        input_params:
            text: string
        output_params: text with all emojis removed
        """
    return emoji.replace_emoji(text,replace='')

#create a function to remove punctuation marks

def remove_punctuation(text):

    """ Function_name: punctuation_remover
        input_params:
            text: string
        output_params: text with all punctuation marks removed
        """
    punctuation_marks = string.punctuation
    return text.translate(str.maketrans('', '', punctuation_marks))

#create function emove url & user

def remove_url_user(text):

    """ Function_name: url_remover
        input_params:
            text: string
        output_params: text with all 'urls' & 'user' removed
        """
    text_1 = text.replace('url','')
    return text_1.replace('user', '')

#convert word to numbers
#nltk_pos_tag for some reason tags certain words as CD e.g girl, so I have to use a tray & except block to handle this

def convert_to_numeric(lists):

    """ Function_name: word2number
        input_params:
            lists: Python list containing strings
        output_params: Python list with all numbers in words converted to digits
        """

    length = len(lists)
    tags = nltk.pos_tag(lists)
    new_list = []
    for i in range(length):
        if tags[i][1] == 'CD':
            try:
                new_list.append(w2n.word_to_num(lists[i]))
            except:
                new_list.append(lists[i])
        else:
            new_list.append(lists[i])
    return new_list

#function to remove all digits

def remove_numbers(lists):

    """ Function_name: digit_remover
        input_params:
            lists: Python list containing strings
        output_params: Python list with all digits removed
        """
    new_list = []
    for val in lists:
        if not bool(re.search(r'\d', str(val))):
            new_list.append(val)
    return new_list

#function to remove stop words

def remove_stop_words(lists):

    """ Function_name: stop_words_remover
        input_params:
            lists: Python list containing strings
        output_params: Python list with all stop words removed
        """

    stoplist = set(stopwords.words('english'))
    new_list = []
    for word in lists:
        if word not in stoplist:
            new_list.append(word)
    return new_list

#function to lemmatize words

def lemmatize_words(lists):

    """ Function_name: lemmatizer
        input_params:
            lists: Python list containing strings
        output_params: Python list with all words lemmatized
        """
    lemmatizer = WordNetLemmatizer()
    new_list = []
    for word in lists:
        new_list.append(lemmatizer.lemmatize(word))
    return new_list

# function to join elements in the list back into a sentence

def rejoin(lists):

    """ Function_name: sentence_maker
        input_params:
            lists: Python list containing strings
        output_params: Python string
        """

    return " ".join(lists)

#create a pipeline unction to apply all the other functions

def text_processing_pipeline(text):

    new_text = contractions.fix(text)
    new_text = lowercase(new_text)
    new_text = remove_emoji(new_text)
    new_text = remove_punctuation(new_text)
    new_text = remove_url_user(new_text)
    new_text = word_tokenize(new_text)
    new_text = convert_to_numeric(new_text)
    new_text = remove_numbers(new_text)
    new_text = remove_stop_words(new_text)
    new_text = lemmatize_words(new_text)
    new_text = rejoin(new_text)

    return new_text

def get_bert_token(input_data):

    """Converts texts to the tokens for bert"""
    token = bert_tokenizer(input_data)

    return token
