import utils as UT
import model
import argparse
import os
from tensorflow.keras.models import load_model
from train import * 


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testPath", required=True,
	          help="path to test file", default="flickr8k_text/Flickr_8k.testImages.txt")

args = vars(ap.parse_args())


model = load_model('model-ep004-loss3.978-val_loss4.168.h5')
test = UT.load_identifiers(arg['testPath'])
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = UT.load_clean_desc('description.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = UT.load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

UT.evaluate_model(model, test_descriptions, test_features, tokens, max_length)