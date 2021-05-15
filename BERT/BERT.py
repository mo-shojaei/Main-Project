try:
    %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import re
import random
import math
from tensorflow.keras import layers
import bert

df = pd.read_csv(r"C:\Users\mosho\OneDrive\Desktop\reviewslabel.csv")

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', str(sentence))

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', str(sentence))

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', str(sentence))

    return str(sentence)
  
  
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', str(text))

#This part cleans all the text review
reviews = []
sentences = list(df['content'])
for sen in sentences:
    reviews.append(preprocess_text(sen))

#creating BERT tokenizer
BertTokenizer = bert.bert_tokenization.FullTokenizer
#create a BERT embedding layer by importing the BERT model from hub.KerasLayer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
#create a BERT vocabulary file in the form a numpy array
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# set the text to lowercase
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
#pass vocabulary_file and to_lower_case variables to the BertTokenizer object
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

#a function that accepts a single text review and returns the ids of the tokenized words in the review
def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))

#execute the following script to actually tokenize all the reviews in the input dataset
tokenized_reviews = [tokenize_reviews(review) for review in reviews]

#creates a list of lists where each sublist contains tokenized review, the label of the review and the length of the review
y = df['Label']
reviews_with_len = [[review, y[i], len(review)]
                 for i, review in enumerate(tokenized_reviews)]

#sort reviews based on length
reviews_with_len.sort(key=lambda x: x[2])
#remove the length attribute from all the reviews
sorted_reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in reviews_with_len]
#convert the sorted dataset into a TensorFlow 2.0-compliant input dataset shape
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))

#pad our dataset for each batch
BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

#divide the dataset into test and training sets
TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

#the model
class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3) 
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output
      
#define the values for the hyper parameters of the model      
VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2

DROPOUT_RATE = 0.2

NB_EPOCHS = 5

#create an object of the TEXT_MODEL class and pass the hyper paramters values that defined in the last step to the constructor of the TEXT_MODEL class
text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)

#compiles the model
if OUTPUT_CLASSES == 2:
    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])
    
#training    
text_model.fit(train_data, epochs=NB_EPOCHS)

#evaluation
results = text_model.evaluate(train_data)
print(results)
