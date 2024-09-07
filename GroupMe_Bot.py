import requests
import json
from flask import Flask, request
from collections import deque, defaultdict
from time import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

BOT_ID = "Place bot id here"
BOT_NAME = "Nike-Zeus"
API_ROOT = 'https://api.groupme.com/v3/'
POST_URL = "https://api.groupme.com/v3/bots/post"
REMOVE_MEMBER_URL = "https://api.groupme.com/v3/groups/{group_id}/members/{member_id}?token={access_token}"
DELETE_MESSAGE_URL = "https://api.groupme.com/v3/groups/{group_id}/messages/{message_id}?token={access_token}"
access_token = "Place token here"  

app = Flask(__name__)

MESSAGE_CACHE_SIZE = 10
message_cache = defaultdict(lambda: deque(maxlen=MESSAGE_CACHE_SIZE))

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

selling_keywords = ['sell', 'selling', 'sale', 'sold', 'vending', 'trading', 'dealing']
ticket_keywords = ['ticket', 'tickets', 'admission', 'pass', 'entry']
concert_keywords = ['concert', 'show', 'performance', 'gig', 'event']
flagged_words = ['dm', 'messag', 'direct', 'contact']

keyword_regex = re.compile(r'\b(' + '|'.join(selling_keywords + ticket_keywords + concert_keywords + flagged_words) + r')\b', re.IGNORECASE)

RATE_LIMIT_WINDOW = 60  # Time window in seconds
RATE_LIMIT_COUNT = 25  # Maximum number of messages allowed within the time window

user_message_counts = defaultdict(list)

def load_training_data(file_path):
    training_data = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            label = row[0]
            message = row[1]
            training_data.append((message, label))
    return training_data

csv_file_path = 'Place Training data here
training_data = load_training_data(csv_file_path)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

# Preprocess the training data
X = [preprocess_text(text) for text, _ in training_data]
y = [label for _, label in training_data]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier during startup
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train_tfidf, y_train)

def classify_message(message):
    preprocessed_message = preprocess_text(message)
    tfidf_feature_vector = vectorizer.transform([preprocessed_message])
    spam_probability = svm_classifier.predict_proba(tfidf_feature_vector)[0][1]
    return spam_probability

def send_message(message):
    data = {
        "bot_id": BOT_ID,
        "text": message
    }
    print(f"Attempting to send message: {message}")
    try:
        response = requests.post(POST_URL, json=data)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        response.raise_for_status()
        print(f"Successfully sent message: {message}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending the message: {e}")
        print(f"Response content: {e.response.text if e.response else 'No response'}")

def is_duplicate_message(user_id, message):
    user_cache = message_cache[user_id]
    return any(cached_msg['text'] == message for cached_msg in user_cache)

def add_to_cache(user_id, message):
    user_cache = message_cache[user_id]
    user_cache.append({'text': message, 'time': time()})

def is_spam(user_id, message):
    user_cache = message_cache[user_id]
    spam_count = sum(keyword_regex.search(cached_msg['text']) is not None for cached_msg in user_cache)
    return spam_count > 3

def is_rate_limited(user_id):
    current_time = time()
    user_message_counts[user_id] = [t for t in user_message_counts[user_id] if current_time - t <= RATE_LIMIT_WINDOW]
    if len(user_message_counts[user_id]) >= RATE_LIMIT_COUNT:
        return True
    user_message_counts[user_id].append(current_time)
    return False
def get_memberships(group_id):
    url = f'{API_ROOT}groups/{group_id}'
    params = {'token': access_token}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['response']['members']
    else:
        print(f"Failed to retrieve memberships for group {group_id}. Status code: {response.status_code}")
        return []

def get_membership_id(group_id, user_id):
    memberships = get_memberships(group_id)
    for membership in memberships:
        if membership['user_id'] == user_id:
            return membership['id']
    return None

def remove_member(group_id, membership_id):
    url = f'{API_ROOT}groups/{group_id}/members/{membership_id}/remove'
    params = {'token': access_token}
    response = requests.post(url, params=params)
    if response.status_code == 200:
        print(f"Successfully removed member {membership_id} from group {group_id}")
        return True
    else:
        print(f"Failed to remove member {membership_id} from group {group_id}. Status code: {response.status_code}")
        return False

def delete_message(group_id, message_id):
    url = f'{API_ROOT}conversations/{group_id}/messages/{message_id}'
    params = {'token': access_token}
    response = requests.delete(url, params=params)
    if response.status_code == 204:
        print(f"Successfully deleted message {message_id} from group {group_id}")
        return True
    else:
        print(f"Failed to delete message {message_id} from group {group_id}. Status code: {response.status_code}")
        return False

def kick_user(group_id, user_id):
    membership_id = get_membership_id(group_id, user_id)
    if membership_id:
        return remove_member(group_id, membership_id)
    else:
        print(f"User {user_id} not found in group {group_id}")
        return False

def handle_message(message, user_id, group_id, message_id, sender_id):
    print(f"Handling message: {message}")
    
    if is_rate_limited(user_id):
        print("User exceeded rate limit")
        send_message("You're sending messages too quickly. Please slow down.")
        return
    
    if BOT_NAME.lower() in message.lower() and "about" in message.lower():
        print("Responding with bot description")
        send_message("I'm a bot that leverages NLP techniques and machine learning to understand the content of messages and determine if they are related to specific topics (selling, tickets, concerts) or potentially spam/fraudulent. It helps in identifying and flagging messages that might require special attention or assistance.\n\nBy using NLTK, text preprocessing, and a Support Vector Machine (SVM) classifier with TF-IDF features, I can handle variations in word forms, filter out irrelevant words, and classify messages based on their content. The keyword matching and counting mechanism, along with the trained SVM classifier, allows me to determine the relevance and potential spam/fraudulent nature of a message.")
        return
    
    spam_probability = classify_message(message)
    
    if is_spam(user_id, message) or spam_probability > 0.5:
        print("Message flagged as spam")
        if sender_id != BOT_ID:
            send_message(f"[ALERT] This message has been flagged as spam or fraudulent with a probability of {spam_probability:.2%}. The user will be removed from the group, and the message will be deleted.")
            kick_user(group_id, user_id)
            delete_message(group_id, message_id)
        else:
            print("Skipping deletion of bot's own message")
    
    elif keyword_regex.search(message):
        print("Message flagged based on keyword matches")
        print("Sending response")
        send_message(f"This message has been flagged as potentially related to selling, tickets, or concerts. It has a {spam_probability:.2%} probability of being spam or fraudulent. How can I assist you with that?")
        add_to_cache(user_id, message)
    else:
        print("Message not flagged")
        if not is_duplicate_message(user_id, message):
            print("Sending generic response")
            send_message(f"I received your message: '{message}'. It has a {spam_probability:.2%} probability of being spam or fraudulent. How can I assist you?")
            add_to_cache(user_id, message)
        else:
            print("Duplicate message found in user's cache, ignoring generic response")

@app.route('/', methods=['POST', 'GET', 'HEAD'])
def root():
    print(f"Received {request.method} request to /")
    
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            print(f"Parsed JSON data: {json.dumps(data, indent=2)}")
            
            if data and 'name' in data and 'text' in data and 'user_id' in data and 'group_id' in data and 'id' in data and 'sender_id' in data:
                message_text = data['text']
                sender_name = data['name']
                user_id = data['user_id']
                group_id = data['group_id']
                message_id = data['id']
                sender_id = data['sender_id']
                
                if sender_name.lower() != BOT_NAME.lower():
                    print(f"Processing new message: {message_text}")
                    handle_message(message_text, user_id, group_id, message_id, sender_id)
                else:
                    print("Ignoring bot message")
            else:
                print("Received incomplete data in webhook")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON data: {e}")
        except Exception as e:
            print(f"An error occurred while processing the webhook: {e}")
    
    return "OK", 200

if __name__ == "__main__":
    print(f"Starting server with BOT_ID: {BOT_ID}")
    send_message("Nike-Zues is starting up!")
    app.run(debug=True, port=5000)