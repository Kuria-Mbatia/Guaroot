# Nike-Zeus
![A Nike Zeus A missile being test launched at White Sands illustrates the long wings and narrow fuselage that carried over from Hercules.](http://www.ninfinger.org/models/scaleroc/Nike-Zeus%20A%20antimissile/nza%2001.jpg)

[Wiki Link on Nike-Zeus Missile Defense Platform](https://en.wikipedia.org/wiki/Nike_Zeus#Nike-X)

# Where does the name Nike-Zues come from?
>"Nike Zeus was an anti-ballistic missile (ABM) system developed by the United States Army during the late 1950s and early 1960s that was designed to destroy incoming Soviet intercontinental ballistic missile warheads before they could hit their targets"

### In our case the Soviet ICBM's are spam bots infiltrating GroupMe servers posing as students

## Why even impliment such a system when a simple key word searching bot is good enough?
>Good enough is alright but we want the ability for a system to learn and change, even a flawed adaptive system is better than none at all.

### This GroupMe bot utilizes Natural Language Processing (NLP) techniques and machine learning to analyze messages, detect spam, and provide intelligent responses in a GroupMe chat.

## What the Code Does
### The code implements a Flask web application that processes incoming GroupMe messages. Here's a breakdown of its main functions:

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

### Probability Scoring on messages 
>The bot assigns spam probability scores using a Support Vector Machine (SVM) classifier. It uses the ```classify_message()``` function, which preprocesses the input text and uses the pre-trained SVM model to predict the probability of a message being spam. Current efficieny rates are between 97.648% - 98.8220% accurate on assigning a probability score on if a message is spam or not.

## NLP Techniques and Machine Learning
>Text preprocessing: Tokenization, stopword removal, and lemmatization using NLTK [Source](https://www.kaggle.com/code/awadhi123/text-preprocessing-using-nltk)
>Feature extraction: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization [Source](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
>Machine learning: Support Vector Machine (SVM) classifier for spam detection [Source](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)
>Keyword matching: Regular expressions for detecting specific patterns in messages (Source)[https://towardsdatascience.com/keyword-extraction-process-in-python-with-natural-language-processing-nlp-d769a9069d5c]




