# Natural-Language-Processing-with-TensorFlow
A series of different Natural Language Processing modellings experiments with various models to predict a tweet is disaster or not



<p align="center">
  <img src="https://help.twitter.com/content/dam/help-twitter/brand/logo.png" width=200 />
</p>


## Table of Content
  * [The Problem](#the-problem)
  * [Goal](#goal)
  * [Project Main Steps](#project-main-steps)
  * [Modeling](#modeling)
    * [Demo TensorBoard](#demo-tensorboard)
  * [Conclusion](#conclusion)
  * [Software and Libraries](#software-and-libraries)


## The Problem
Twitter has become an important communication channel in times of emergency. The ubiquitousness of smartphones enables people to announce an emergency they're observing in real-time. Because of this, more agencies are interested in programmatically monitoring Twitter (e.g., disaster relief organizations and news agencies). But, it's not always clear whether a person's words are actually announcing a disaster. Take this example:

@SonofLiberty357 all illuminated by the brightly burning buildings all around the town! ----

The author explicitly uses the word "burning" but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it's less clear to a machine.


## Goal
The problem to be solved by this capstone project is how to identify which tweets are about "real disasters" and which ones aren't.T his project will use a data science approach to build a machine learning classifier model to predict which tweets are about 'real disasters' and which one's aren't.

## Project Main Steps:

- Downloading a text dataset
- Visualizing text data
- Converting text into numbers using tokenization
- Turning our tokenized text into an embedding
- Modeling a text dataset
  - Starting with a baseline (TF-IDF)
- Building several deep learning text models
  - Dense, LSTM, GRU, Conv1D, Transfer learning
- Comparing the performance of each of our models
- Combining our models into an ensemble
- Saving and loading a trained model
- Find the most wrong predictions

## Modeling

```
Text -> turn into numbers -> build a model -> train the model to find patterns -> use patterns (make predictions)
```
### Demo TensorBoard

#### [Clik Here!](https://tensorboard.dev/experiment/uZkruxrDTbO0mIVGGjkO6A/#scalars&run=Conv1D%2F20220107-013633%2Ftrain)
[<img target="_blankk" src="https://github.com/docum5/Natural-Language-Processing-with-TensorFlow/blob/main/Screen%20Shot%202022-01-07%20at%2011.20.38.png?raw=true" >](https://tensorboard.dev/experiment/uZkruxrDTbO0mIVGGjkO6A/#scalars&run=Conv1D%2F20220107-013633%2Ftrain)


| Algorithm               | Accuracy | Precision | Recall | F1   |
|-------------------------|----------|-----------|--------|------|
| Naive Bayes             | 0.79     | 0.811     | 0.79   | 0.78 |
| ANN(Simple dense)       | 0.78     | 0.79      | 0.78   | 0.78 |
| lstm                    | 0.76     | 0.76      | 0.76   | 0.76 |
| Gru                     | 0.77     | 0.77      | 0.77   | 0.76 |
| bidirectional           | 0.73     | 0.73      | 0.73   | 0.73 |
| conv1d                  | 0.78     | 0.78      | 0.78   | 0.78 |
| tf_hub_sentence_encoder | 0.81     | 0.81      | 0.81   | 0.80 |
| Models Ensembling       | 0.78     | 0.78      | 0.78   | 0.78 |


Comparing the Performance of Each of Our Models           | Comparing the Performance by F1-score
:-------------------------:|:-------------------------:
![](https://github.com/docum5/Natural-Language-Processing-with-TensorFlow/blob/main/Screen%20Shot%202022-01-07%20at%2011.30.23.png?raw=true)  | ![](https://github.com/docum5/Natural-Language-Processing-with-TensorFlow/blob/main/Screen%20Shot%202022-01-07%20at%2011.29.53.png?raw=true)


## Conclusion
In this capstone project, I took a Kaggle challenge to classify tweets into disaster tweets in real or not?. First, I have analyzed and explored all the provided tweets data to visualize the statistical and other properties of the presented data. Next, I performed some exploratory analysis of the data to check the type of the data, whether there are unwanted features and if features have missing data. Based on the analysis, I decided to drop the "location" and "keyword" column since it has most of the data missing and really has no effect on the classification of tweets. The 'text' columns are all text data along with alphanumeric, special characters, and embedded URLs.The 'text' column data needs to be cleaned, pre-processed and vectorized before using a machine-learning algorithm to classify the tweets. After pre-processing the train and test data, the data was vectorized using CountVectorizer and TFIDF features. Then it was split into training and validation data, and then various classifiers were fit on the data, and predictions were made. Out of all classifiers tested, tf_hub_sentence_encoder(using pre-trained embedding universal sentence encoder) performed the best with the test accuracy of 81,1%. The second best choice model is Naive Bayes, with a test accuracy of 79,2%.


## Software and Libraries
This project uses the following software and Python libraries:



![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/2560px-Scikit_learn_logo_small.svg.png" width=100>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/440px-NumPy_logo_2020.svg.png" width=150>](https://numpy.org/) [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1024px-Pandas_logo.svg.png" width=200>](https://pandas.pydata.org/docs/getting_started/index.html) [<img target="_blank" src="https://camo.githubusercontent.com/aeb4f612bd9b40d81c62fcbebd6db44a5d4344b8b962be0138817e18c9c06963/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74665f6c6f676f5f686f72697a6f6e74616c2e706e67" width=200>](https://www.tensorflow.org/) [<img target="_blank" src="https://matplotlib.org/stable/_static/logo2.svg" width=100 height=50>](https://matplotlib.org/)



