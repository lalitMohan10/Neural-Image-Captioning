import os 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add





def getmodel():
    #load model
    model = VGG16()
    #remove output layer
    model.layers.pop()
    model = Model(inputs= model.inputs , outputs= model.layers[-1].output)
    return model




def define_model(vocab_size, max_length):
    #encoder1
    input1 = Input(shape=(4096,), name='Encoder1')
    en1    = Dropout(0.5)(input1)
    en2    = Dense(256,activation='relu' )(en1)
    
    #encoder2
    input2 = Input(shape=(max_length,), name='Encoder2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(input2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    #decoder
    merge = add([en2,se3])
    decoder1 = Dense(256, activation="relu")(merge)
    output = Dense(vocab_size, activation='softmax')(decoder1)
    
    #put it together
    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    plot_model(model, to_file='model.png', show_shapes=True)
    print(model.summary())
    
    return model