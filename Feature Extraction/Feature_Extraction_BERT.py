#import the libraries we needed
import math
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import Model 
import bert
import pandas as pd

#define the bert tokenizer
FullTokenizer = bert.bert_tokenization.FullTokenizer

#bulding the model
#define maximum length of a sequence after tokenizing
max_seq_length = 128 
#tokenizer converts tokens using vocab file
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
#1 for useful tokens, 0 for padding
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
#for 2 text training: 0 for the first one, 1 for the second one
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
#import the bert layer we use 
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
#pooled_output of shape [batch_size, 768] with representations for the entire input sequences
#sequence_output of shape [batch_size, max_seq_length, 768] with representations for each input token 
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

#define the model
model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

#get the masks that we defined from tokenized review
def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

#get the segments that we defined from tokenized review
def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

#get the ids that we defined from tokenized review
def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

#Import tokenizer using the original vocab file and lowercase all the words

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

#Import the data set
df = pd.read_csv('data/reviews.csv')

word_list = []
word_list_dict = {}
for idx, row  in df.iterrows():
    words = tokenizer.tokenize(row['content'])
    words = normalize(words)
    word_list.append(words)
    word_list_dict[idx] = word_list

for words in world_list_dict[idx]:
input_ids = get_ids(words, tokenizer, max_seq_length)
input_masks = get_masks(words, max_seq_length)
input_segments = get_segments(words, max_seq_length)
