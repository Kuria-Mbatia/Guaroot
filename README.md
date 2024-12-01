
<H1 align="center">Guaroot</H1>
<H3 align="center">"Root out the bad stuff🍂"</H3>
<p align="center">
  <img src="https://github.com/Kuria-Mbatia/Nike-Zeus/blob/main/Guaroot%20Images/file%20(1).jpg" />
</p>

<h2>Formerly known as Nike-Zeus</h2>
[Wiki Link on Nike-Zeus Missile Defense Platform](https://en.wikipedia.org/wiki/Nike_Zeus])

# Where does the name Nike-Zues come from?
>"Nike Zeus was an anti-ballistic missile (ABM) system developed by the United States Army during the late 1950s and early 1960s that was designed to destroy incoming Soviet intercontinental ballistic missile warheads before they could hit their targets"

In our case the Soviet ICBM's are spam bots infiltrating GroupMe servers posing as students. Nike-Zeus is a sophisticated GroupMe bot that employs advanced Natural Language Processing (NLP) techniques and machine learning to analyze messages, detect spam, and provide intelligent moderation capabilities in a GroupMe chat environment. Inspired by the Nike Zeus anti-ballistic missile system designed to intercept Soviet ICBMs, this bot aims to intercept and neutralize spam messages infiltrating GroupMe servers.

## Why even impliment such a system when a simple key word searching bot is good enough?
>Good enough is alright, but we want the ability for a system to learn and change, while a simple keyword-searching bot might suffice for basic spam detection, Nike-Zeus takes it a step further by incorporating an adaptive learning system. Even a flawed adaptive system is preferable to none at all, as it allows for continuous improvement and refinement over time

### This GroupMe bot utilizes Natural Language Processing (NLP) techniques and machine learning to analyze messages, detect spam, and provide intelligent responses in a GroupMe chat.

## What the Code Does
### Spam and Fraudulent Message Detection:
>Nike-Zeus employs an SVM (Support Vector Machine) classifier trained on a dataset stored in a CSV file (spam.csv) to identify spam and potentially fraudulent messages. It utilizes TF-IDF (Term Frequency-Inverse Document Frequency) features to represent messages and assigns spam probability scores.

### Keyword-based Flagging: 
>The bot checks messages against predefined sets of keywords related to selling, tickets, concerts, and other flagged words. Messages containing these keywords are flagged and added to the user's message cache for further analysis.
### Rate Limiting:
>To prevent users from sending messages too quickly, Nike-Zeus implements rate limiting. If a user exceeds the defined rate limit, the bot sends a warning message to maintain a controlled chat environment.
### User and Message Management:
>When a message is flagged as spam or fraudulent, Nike-Zeus has the ability to remove the offending user from the group and delete the flagged message using the GroupMe API.
### Message Caching and Duplicate Detection:
>The bot maintains a cache of recently sent messages for each user and checks for duplicates to avoid processing the same message multiple times, optimizing its efficiency.


### Model Retraining:
>Nike-Zeus continuously updates its training dataset (spam.csv) with newly flagged spam messages and triggers a model retraining process in a separate thread. This ensures that the classifier stays up to date with the latest spam patterns and maintains its effectiveness over time.
>An incrimental learning approach (SGD) will be used later on for better resource management rather than frequent retraining.  

# How It Works
## Nike-Zeus is a Flask web application that listens for incoming POST requests from the GroupMe webhook. It processes messages through a series of steps, including text preprocessing, spam classification, keyword matching, action taking, rate limiting, and response generation.


# Code Structure
The Nike-Zeus codebase is organized into several key functions:
>```
>load_training_data(): Loads spam classification training data from a CSV file.
>preprocess_text(): Preprocesses text by tokenizing, removing stopwords, and lemmatizing.
>classify_message(): Assigns a spam probability score to a message using a trained SVM classifier.
>send_message(): Sends a message to the GroupMe chat.
>is_duplicate_message(): Checks if a message is a duplicate.
>is_spam(): Determines if a message is spam based on keyword matching and message history.
>is_rate_limited(): Implements rate limiting for user messages.
>handle_message(): Main function for processing incoming messages, detecting spam, and generating responses.
>```

# NLP Techniques and Machine Learning
Nike-Zeus utilizes a combination of NLP techniques and machine learning algorithms:
Text Preprocessing: Tokenization, stopword removal, and lemmatization using NLTK.
Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
Machine Learning: Support Vector Machine (SVM) classifier for spam detection.
Keyword Matching: Regular expressions for detecting specific patterns in messages.


### Probability Scoring on messages 
>The bot assigns spam probability scores using a Support Vector Machine (SVM) classifier. It uses the ```classify_message()``` function, which preprocesses the input text and uses the pre-trained SVM model to predict the probability of a message being spam. Current efficieny rates are between 97.648% - 98.8220% accurate on assigning a probability score on if a message is spam or not.

## NLP Techniques and Machine Learning
>Text preprocessing: Tokenization, stopword removal, and lemmatization using NLTK [Source](https://www.kaggle.com/code/awadhi123/text-preprocessing-using-nltk)
>
>Feature extraction: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization [Source](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
>
>Machine learning: Support Vector Machine (SVM) classifier for spam detection [Source](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)
>
>Keyword matching: Regular expressions for detecting specific patterns in messages [Source](https://towardsdatascience.com/keyword-extraction-process-in-python-with-natural-language-processing-nlp-d769a9069d5c)




