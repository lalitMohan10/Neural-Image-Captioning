import os
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from nltk.translate.bleu_score import corpus_bleu
import model as mm
from tqdm import tqdm
from pickle import load, dump





def load_Doc(path):
    file = open(path, 'r')
    text = file.read()
    file.close()
    return text


def load_Desc(doc):
    mapping= dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line)<2:
            continue
        #get imageid
        imageid = tokens[0].split('.')[0]
        #get description
        desc = " ".join(tokens[1:])
        #check if imageid exist
        if imageid not in mapping:
            mapping[imageid]= list()
        mapping[imageid].append(desc)
    return mapping



def load_identifiers(path):
    
    doc = load_Doc(path)
    dataset = list()
    for line in doc.split("\n"):
        if len(line)<1:
            continue
        identifiers = line.split('.')[0]
        dataset.append(identifiers)
    return set(dataset)


def cleaning(description):
    # prepare translation table for removing punctuations
    table = str.maketrans('', '', string.punctuation)

    for imageid, desc_list in description.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] =  ' '.join(desc)


def load_clean_desc(filename, dataset):
    
    doc = load_Doc(filename)
    descriptions = dict()
    for line in doc.split("\n"):
        tokens = line.split()
        imageid, imagedesc = tokens[0], tokens[1:]
        if imageid in dataset:
        #skip image if not in dataset
            if imageid not in descriptions:
                descriptions[imageid] = list()
            desc = 'startseq' + ' '.join(imagedesc) + 'stopseq'
            descriptions[imageid].append(desc)
    return descriptions



def to_vocabolary(desc):
    vocab_set = set()
    for key in desc.keys():
        [vocab_set.update(d) for d in desc[key]]
    return vocab_set


def save_description(desc, filename):
    lines = list()
    for keys, des in desc.items():
        for d in des:
            lines.append(keys+' '+d)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def to_line(descriptions):
    lines =[]
    for keys in descriptions.keys():
        [lines.append(d) for d in descriptions[keys]]
    return lines



def load_tokens(description):
    desc = to_line(description)
    token = Tokenizer()
    token.fit_on_texts(desc)
    return token



def max_length(description):
    lines = to_line(description)
    return max(len(d.split()) for d in lines)


def create_sequence(tokeniser, maxlength, photo, description, vocab_size):
    X1,X2,Y = list(),list(),list()
    
    # walk through each image identifier
    for key, desc_list in description.items():
        for d in desc_list:
            seq = tokeniser.texts_to_sequences([d])[0]
            #split each seq into X,Y pair
            for i in range(1,len(seq)):
                input_seq, out_seq = seq[:i],seq[i]
                #pad the input sequence
                in_seq = pad_sequences([input_seq], maxlen=max_length)[0]
                #encode the output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                #store all values
                X1.append(photo[key][0])
                X2.append(in_seq)
                Y.append(out_seq)
        
    return np.array(X1),np.array(X2),np.array(Y)




def extract_features(directory):
    
    #feature dict
    features = dict()
    print("[INFO] loading model.....")
    model = mm.getmodel()
    print("[INFO] Model loaded.....")
    for name in tqdm(os.listdir(directory)):
        
        image_id = name.split('.')[0]
        #get filename
        filename = os.path.join(directory, name)
        #load image
        image = load_img(filename, target_size=(224,224))
        #convert to array
        image = img_to_array(image)
        #reshape image to input size to the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        #prepare image for VGG model
        image = preprocess_input(image)
        
        #get features
        feature = model.predict(image, verbose=0)
        #store features
        features[image_id] = feature
        
    return features



def dumpfile(output, file):
    return dump(file, open(output, 'wb'))


def load_photo_features(filename, dataset):
    #load features from pickle file
    feat = load(open(filename, 'rb'))
    features = {k:feat[k] for k in dataset}
    return features


#Functions to evaluate the model

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
  actual, predicted = list(),list()

  for key, desc_list in descriptions.items():

    yHat = gen_desc(model, tokenizer, photos[key], max_length)

    reference = [d.split() for d in desc_list]
    actual.append(reference)
    predicted.append(yHat.split())
  
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))



def gen_desc(model, tokenizer, photo, max_length):
  in_text ='startseq'

  for i in range(max_length):

    seq = tokenizer.texts_to_sequences([in_text])[0]
    seq = pad_sequences([seq], maxlen=max_length)

    yHat = model.predict([photo,seq],verbose=0)
    yHat = np.argmax(yHat)

    word = word_for_id(yHat, tokenizer)
    if word is None:
      break
    in_text+= " " + word
    if word == 'endseq':
      break
  return in_text



def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index== integer:
      return word
  return None