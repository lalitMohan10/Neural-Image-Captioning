import argparse
import utils as UT
from pickle import load
import os
import model
from tensorflow.keras.callbacks import ModelCheckpoint




ap = argparse.ArgumentParser()
ap.add_argument("-t", "--textPath", required=True,
	          help="path to text file", default="flickr8k_text/Flickr8k.lemma.token.txt")

ap.add_argument("-tr", "--trainPath", required=True,
	          help="path to train file", default="flickr8k_text/Flickr_8k.trainImages.txt")

ap.add_argument("-feat", "--features", required=True,
	          help="path to features file", default="Features/features.pkl")

ap.add_argument("-val", "--devPath", required=True,
	          help="path to validation file", default="flickr8k_text/Flickr_8k.devImages.txt")


args = vars(ap.parse_args())



#Load the text File
doc = UT.load_Doc(args['textPath'])
#load descriptions from textfile
description = UT.load_Desc(doc)
#clean the descriptions
UT.cleaning(description)

#save the description
UT.save_description(description, "description.txt")


#check if pickle file exist
if os.path.isfile('features.pkl'):
	features = load(open('Features/features.pkl', 'rb'))
else:
	features = UT.extract_features(arg['features'])




#Load Training data
train = UT.load_identifiers(arg['trainPath'])
print('Dataset Train: ', len(train))

"====================================================================="

#load train set description
train_descriptions = UT.load_clean_desc('description.txt', train)
print("Descriptions Train: ", len(train_descriptions))

"======================================================================"

train_features = UT.load_photo_features('features.pkl', train)
print("train features:", len(train_features))

"======================================================================="

# Get tokens 
tokens = UT.load_tokens(train_descriptions)
vocab  = len(tokens.word_index)+1
print('Vocab Size:', vocab)



max_length = UT.max_length(train_descriptions)
print('Description Length:', max_length)
# prepare sequences
X1train, X2train, ytrain = UT.create_sequence(tokens, max_length,features,train_descriptions, vocab)
print('Size of sequence',len(X2train))



# TIME FOR LOAD VALIDATION DATASET
print("[INFO] Load Val data.......")
test = UT.load_identifiers(args['devPath'])
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = UT.load_clean_desc('description.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = UT.load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = UT.create_sequence(tokens, max_length, test_features,test_descriptions, vocab)
print('Size of text Sequence:', len(X2test))
print("[INFO] Val Data Loaded.......")



print("[INFO] MODEL LOADING.....")
# define the model
model = mm.define_model(vocab, max_length)
print("[INFO] MODEL LOADED......")
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


print('[INFO] Model fitting.... ')
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))