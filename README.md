# Natural-Language-Processing-with-TensorFlow
A series of different Natural Language Processing modellings experiments with various models to predict a tweet is disaster or not

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


## Conclusion
In this capstone project, I took a Kaggle challenge to classify tweets into disaster tweets in real or not?. First, I have analyzed and explored all the provided tweets data to visualize the statistical and other properties of the presented data. Next, I performed some exploratory analysis of the data to check the type of the data, whether there are unwanted features and if features have missing data. Based on the analysis, I decided to drop the "location" and "keyword" column since it has most of the data missing and really has no effect on the classification of tweets. The 'text' columns are all text data along with alphanumeric, special characters, and embedded URLs.The 'text' column data needs to be cleaned, pre-processed and vectorized before using a machine-learning algorithm to classify the tweets. After pre-processing the train and test data, the data was vectorized using CountVectorizer and TFIDF features. Then it was split into training and validation data, and then various classifiers were fit on the data, and predictions were made. Out of all classifiers tested, tf_hub_sentence_encoder(using pre-trained embedding universal sentence encoder) performed the best with the test accuracy of 81,1%. The second best choice model is Naive Bayes, with a test accuracy of 79,2%.

