import math

import h5py
import numpy as np
import pandas as pd
import pickle
import os, sys
from collections import Counter, defaultdict
import  json
import argparse
# import keras
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Lambda, LSTM, TimeDistributed, Masking, Bidirectional
from tensorflow.keras.layers import Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
###################################################################################################################################

# Hyperparams
scalar = MinMaxScaler(feature_range=(-1, 1))

class DataHelper:

    TARGET_AUDIO_ID = 1
    TEXT_BERT_ID = 2
    TARGET_VIDEO_ID = 3

    def __init__(self, train_input, train_output, test_input, test_output, dataLoader):
        self.dataLoader = dataLoader

        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output

    def getData(self, ID=None, mode=None, error_message=None):

        if mode == "train":
            return [instance[ID] for instance in self.train_input]
        elif mode == "test":
            return [instance[ID] for instance in self.test_input]
        else:
            print(error_message)
            exit()

    def getTargetBertFeatures(self, mode=None):

        utterances = self.getData(self.TEXT_BERT_ID, mode,
                                  "Set mode properly for vectorizeUtterance method() : mode = train/test")

        return utterances


    def getTargetAudioPool(self, mode=None):

        audio = self.getData(self.TARGET_AUDIO_ID, mode,
                             "Set mode properly for TargetAudio method() : mode = train/test")

        # return np.array([np.mean(feature_vector, axis=1) for feature_vector in audio])
        return audio

    #### Video related functions ####

    def getTargetVideoPool(self, mode=None):
        video = self.getData(self.TARGET_VIDEO_ID, mode,
                             "Set mode properly for TargetVideo method() : mode = train/test")

        return video


class Dataloader:

    DATA_PATH_JSON = "./data/result_new.json"
    AUDIO_PICKLE = "./data/audio_feature_55_segmented_32.p"
    BERT_TARGET_EMBEDDINGS = "./data/bert_feature.p"

    def __init__(self,t=False,a=False,v=False):
        self.t = t
        self.a = a
        self.v = v

        dataset_json = json.load(open(self.DATA_PATH_JSON, encoding='utf8'))

        text_bert_embeddings = pickle.load(open(self.BERT_TARGET_EMBEDDINGS, 'rb'))
        audio_features =  pickle.load(open(self.AUDIO_PICKLE, 'rb'))
        # audio_features['10_024'] =  audio_features['10_024'][:,:252]

        temp = []
        for idx , v in audio_features.items():
            for i in v:
                for j in i:
                    if math.isnan(j):
                        temp.append(idx)

        c = set(temp)

        segmented_video_feature = pickle.load(open('./data/segmented_video_feature.p','rb'))
        # segmented_video_feature = None

        self.data_input, self.data_output = [], []

        for idx, ID in enumerate(dataset_json.keys()):
            flag = False
            for ele in c:
                if ID == ele:
                    flag = True
            if flag:
                continue

            self.data_input.append((
                ID,
                audio_features[ID] if audio_features else None,
                text_bert_embeddings[ID] if text_bert_embeddings else None,
                segmented_video_feature[ID] if segmented_video_feature else None,
            ))
            label = self.get_one_hot(int(dataset_json[ID]["humor"]))
            labels = []
            for i in range(32):
                labels.append(label)
            self.data_output.append(labels)

        self.speakerIndependentSplit()

    def getSpeakerIndependent(self):
        '''
        Returns the split indices of speaker independent setting
        '''
        return self.train_ind_SI, self.test_ind_SI
    def getSplit(self, indices):
        '''
        Returns the split comprising of the indices
        '''
        data_input = [self.data_input[ind] for ind in indices]
        data_output = [self.data_output[ind] for ind in indices]
        return data_input, data_output

    def speakerIndependentSplit(self):
        '''
        Prepares split for speaker independent setting
        Train: Fr, TGG, Sa
        Test: TBBT
        '''
        self.train_ind_SI, self.test_ind_SI = [], []
        for ind, data in enumerate(self.data_input):
            if data[0].split('_')[0] == '10':
                self.test_ind_SI.append(ind)
            else:
                self.train_ind_SI.append(ind)


    def train_IO(self):
        train_input, train_output = data.getSplit(self.train_ind_SI)
        test_input, test_output = data.getSplit(self.test_ind_SI)

        datahelper = DataHelper(train_input, train_output, test_input, test_output, data)

        train_input = np.empty((len(train_input),32, 0))
        test_input = np.empty((len(test_input), 32,0))

        if self.t == True:
            train_input = np.concatenate([train_input,np.array(datahelper.getTargetBertFeatures(mode='train'))],axis=-1)
            test_input = np.concatenate([test_input,np.array(datahelper.getTargetBertFeatures(mode='test'))],axis=-1)

        if self.a == True:
            train_input = np.concatenate([train_input, np.array(datahelper.getTargetAudioPool(mode='train'))], axis=-1)
            test_input = np.concatenate([test_input, np.array(datahelper.getTargetAudioPool(mode='test'))], axis=-1)

        if self.v == True:
            train_input = np.concatenate([train_input,np.array(datahelper.getTargetVideoPool(mode='train'))],axis=-1)
            test_input = np.concatenate([test_input,np.array(datahelper.getTargetVideoPool(mode='test'))],axis=-1)

        # cut_length = train_input.shape[0]
        # vec = np.concatenate([train_input,test_input],axis=0)
        # x,y,z = vec.shape
        # vec = np.reshape(vec,(x,-1))
        # vec = scalar.fit_transform(vec)
        # vec = np.reshape(vec,(x,y,z))
        # train_input = vec[:cut_length] ; test_input = vec[cut_length:]

        return train_input ,train_output ,test_input ,test_output

    def get_audio_model(self,dim):

        self.embedding_dim = dim

        print("Creating Model...")

        inputs = Input(shape=(32, self.embedding_dim), dtype='float32')
        lstm = Bidirectional(LSTM(300, activation='tanh', return_sequences=True, dropout=0.4))(inputs)
        # lstm = Bidirectional(LSTM(300, activation='tanh', return_sequences=True, dropout=0.4), name="utter")(lstm)

        output = TimeDistributed(Dense(2, activation='softmax'))(lstm)

        model = Model(inputs, output)

        return model

    def get_one_hot(self, label):
        label_arr = [0]*2
        label_arr[label]=1
        return label_arr[:]


def calc_test_result(pred_label, test_label):
    true_label = []
    predicted_label = []

    for i in range(pred_label.shape[0]):
        for j in range(pred_label.shape[1]):
            # if test_mask[i, j] == 1:
            true_label.append(np.argmax(test_label[i, j]))
            predicted_label.append(np.argmax(pred_label[i, j]))

    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=4))
    print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='weighted'))


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    if n == 0:
        n = 1
    return [arr[i:i + n] for i in range(0, len(arr), n)]



if __name__ == '__main__':
    import tensorflow as tf
    data = Dataloader(t=True,a=True,v=True)
    (train_index, test_index) = data.getSpeakerIndependent()
    train_input, train_output, test_input, test_output = data.train_IO()

    train_data = tf.data.Dataset.from_tensor_slices((train_input,train_output))
    train_data = train_data.batch(64,drop_remainder=True)
    # test_data = tf.data.Dataset.from_tensor_slices((test_input,test_output))
    # test_data = test_data.batch(32,drop_remainder=True)

    model = data.get_audio_model(dim = train_input.shape[-1])
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,loss=tf.losses.CategoricalCrossentropy(from_logits=True) ,metrics=['CategoricalAccuracy'])
    model.fit(train_data,
               epochs= 10,
               # batch_size=64,
               # shuffle=True,
               # callbacks=[early_stopping, checkpoint],
               # validation_data= test_data
              )
    # valuation
    test_predict = model.predict(np.array(test_input))
    test_predict_1 = np.array(test_predict)
    calc_test_result(test_predict_1, np.array(test_output))
