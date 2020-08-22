#!/usr/bin/env python
import numpy as np, os, sys, joblib
import joblib
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat



def res_block(X):
  X_shortcut = X
  X_shortcut = keras.layers.MaxPool1D(pool_size=1)(X_shortcut)

  X = keras.layers.BatchNormalization()(X)
  X = keras.layers.Activation("relu")(X)
  X = keras.layers.Dropout(0.2)(X)
  X = keras.layers.Conv1D(filters=12, kernel_size=5, activation="relu", padding="same")(X)
  X = keras.layers.BatchNormalization()(X)
  X = keras.layers.Activation("relu")(X)
  X = keras.layers.Dropout(0.2)(X)
  X = keras.layers.Conv1D(filters=12, kernel_size=5, activation="relu", padding="same")(X)
  X = keras.layers.add([X,X_shortcut])
  return X


def create_model():

    inputA = keras.layers.Input(shape=(5000, 12))
    inputB = keras.layers.Input(shape=(2,))
    # conv block -1
    conv1 = keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(inputA)
    conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
    conv1 = keras.layers.Dropout(rate=0.2)(conv1)
    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
    # conv block -2
    conv2 = keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
    conv2 = tfa.layers.InstanceNormalization()(conv2)
    conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
    conv2 = keras.layers.Dropout(rate=0.2)(conv2)
    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
    # conv block -3
    conv3 = keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
    conv3 = tfa.layers.InstanceNormalization()(conv3)
    conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
    conv3 = keras.layers.Dropout(rate=0.2)(conv3)
    # split for attention
    attention_data = keras.layers.Lambda(lambda x: x[:,:,:256])(conv3)
    attention_softmax = keras.layers.Lambda(lambda x: x[:,:,256:])(conv3)
    # attention mechanism
    attention_softmax = keras.layers.Softmax()(attention_softmax)
    multiply_layer = keras.layers.Multiply()([attention_softmax,attention_data])
    # last layer
    dense_layer = keras.layers.Dense(units=256,activation='sigmoid')(multiply_layer)
    dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
    # output layer
    #flatten_layer = keras.layers.Flatten()(dense_layer)
    #output_layer = keras.layers.Dense(units=24,activation='relu')(flatten_layer)

    output_layer = keras.layers.Flatten()(dense_layer)  

    mod1 = keras.Model(inputs=inputA, outputs=output_layer)



    mod2 = keras.layers.Dense(10, activation="sigmoid")(inputB)
    mod2 = keras.models.Model(inputs=inputB, outputs=mod2)

    combined = keras.layers.concatenate([mod1.output, mod2.output])

    z = keras.layers.Dense(27, activation="sigmoid")(combined)

    model = keras.models.Model(inputs=[mod1.input, mod2.input], outputs=z)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=[tf.keras.metrics.BinaryAccuracy(
            name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'), 
                        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name="AUC",
            dtype=None,
            thresholds=None,
            multi_label=True,
            label_weights=None,
        )])

    return model



def run_12ECG_classifier(data,header_data,loaded_model):
    


    threshold = np.array([0.22119875, 0.19416626, 0.10130766, 0.17083368, 0.18490477,
0.28237364, 0.18305087, 0.23531812, 0.28968052, 0.26430463,
0.10888419, 0.09313463, 0.21084098, 0.32793421, 0.02737948,
0.00189581, 0.11777638, 0.16809073, 0.21648457, 0.26813463,
0.30477378, 0.21385738, 0.46887888, 0.08739123, 0.2023359 ,
0.19311535, 0.20154463])


    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model
    padded_signal = keras.preprocessing.sequence.pad_sequences(data, maxlen=5000, truncating='post',padding="post")
    reshaped_signal = padded_signal.reshape(1,5000,12)

    gender = header_data[14][6:-1]
    age=header_data[13][6:-1]
    if gender == "Male":
        gender = 0
    elif gender == "male":
        gender = 0
    elif gender =="M":
        gender = 0
    elif gender == "Female":
        gender = 1
    elif gender == "female":
        gender = 1
    elif gender == "F":
        gender = 1
    elif gender =="NaN":
        gender = 2

    # Age processing - replace with nicer code later
    if age == "NaN":
        age = -1
    else:
        age = int(age)

    demo_data = np.asarray([age,gender])
    reshaped_demo_data = demo_data.reshape(1,2)

    combined_data = [reshaped_signal,reshaped_demo_data]


    score  = model.predict(combined_data)[0]
    
    binary_prediction = score > threshold
    binary_prediction = binary_prediction * 1
    classes = ['10370003','111975006','164889003','164890007','164909002','164917005','164934002','164947007','17338001',
 '251146004','270492004','284470004','39732003','426177001','426627000','426783006','427084000','427172004','427393009','445118002','47665007','59118001',
 '59931005','63593006','698252002','713426002','713427006']

    return binary_prediction, score, classes

def load_12ECG_model(model_input):
    model = create_model()
    f_out='model.h5'
    filename = os.path.join(model_input,f_out)
    model.load_weights(filename)

    return model
