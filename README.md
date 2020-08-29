# Automatic-Image-Caption-Generation

Caption generation is a challenging artificial intelligence problem where a textual description must be generated for a given photograph.

It requires both methods from computer vision to understand the content of the image and a language model from the field of natural language processing to turn the understanding of the image into words in the right order. Recently, deep learning methods have achieved state-of-the-art results on examples of this problem.

### **Acknowledgement**

I would like to thank **Jason Brownlee** for his wonderful [blog](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/) that help me to learn how to build a `Image Caption Generator`.


## Content:

- [Dependencies](https://github.com/arpitj07/Automatic-Image-Caption-Generation/blob/master/README.md#dependencies)
- [Dataset](https://github.com/arpitj07/Automatic-Image-Caption-Generation/blob/master/README.md#dataset)
- [Helper Functions](https://github.com/arpitj07/Automatic-Image-Caption-Generation/blob/master/README.md#helper-functions)
- [Training](https://github.com/arpitj07/Automatic-Image-Caption-Generation/blob/master/README.md#training)
- [Evaluation](https://github.com/arpitj07/Automatic-Image-Caption-Generation/blob/master/README.md#evaulation)
- [Architechture](https://github.com/arpitj07/Automatic-Image-Caption-Generation/blob/master/README.md#architechture)


### [WARNING]:
This project need high RAM : 32 GB/ 64 GB. Either you can use AWS EC2 instance or Google Collaboratory[The one I used].


### Dependencies:
This project requires a lot of modules and packages. This can be installed from `requirement.txt` file using following command:

```
pip install -r requirements.txt for python 2.x
pip3 install -r requirements.txt for python 3.x
```

### Helper Functions:
All the helper functions needed for this project are in `utility.py` file.

### Dataset:
The dataset can be downloaded from `Kaggle` from [here](https://www.kaggle.com/ming666/flicker8k-dataset). You can use the already created feature file [features extracted from images] located in `Features` folder. Its compressed you need to unzip it.

The dataset consists of 2 files:
1) Images 
2) Description and Image IDs


### Training:
Now comes the training part. To train your model defined in `model.py` we will run the `train.py` file. 
**Remember** training may take longer time to run depends on the congiguration of the machine. Each epoch takes around 15-20 mins.

training need 4 arguments:
1. textPath
2. trainPath
3. devPath
4. features

```
python train.py --textPath /Path to Textfile/ --trainPath /Path to trainfile/  --devPath /Path to valimages/ --features /Path to features/
```

### Evaluation:
After training we will evaluate our model on test dataset. Run the following command:
```
python evaluate.py --testPath /Path to testfile/
```

### Architechture:

The model Architechture:

![MODEL ARCHITECHTURE](https://github.com/arpitj07/Automatic-Image-Caption-Generation/blob/master/Model/model.png)
