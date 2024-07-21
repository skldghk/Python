from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from fuzzywuzzy import fuzz
from urllib.parse import quote_plus
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
# nltk 데이터 다운로드
nltk.download('punkt')
nltk.download('wordnet')

# 모델과 데이터 로드
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = json.loads(open('words.json', encoding='utf-8').read())
classes = json.loads(open('classes.json', encoding='utf-8').read())

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# SQLAlchemy 데이터베이스 URI 설정
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='chat', lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    sender = db.Column(db.String(50), nullable=False)
    text = db.Column(db.Text, nullable=False, default="")
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_best_matching_pattern(sentence, patterns):
    best_match = None
    highest_similarity = 0
    for pattern in patterns:
        similarity = fuzz.ratio(sentence, pattern)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = pattern
    return best_match

def get_response(ints, intents_json, msg):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            best_pattern = get_best_matching_pattern(msg, i['patterns'])
            best_pattern_index = i['patterns'].index(best_pattern)
            if 'detailed_response' in i:
                if tag == 'science':
                    return i['detailed_response'][best_pattern_index]
            return i['responses'][0]
    return "죄송합니다, 이해하지 못했어요. 다시 말씀해주시겠어요?"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("home"))
        else:
            flash("사용자 이름 또는 비밀번호가 틀렸습니다.")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        existing_user = User.query.filter_by(username=username).first()
        if existing_user is None:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for("home"))
        else:
            flash("이미 존재하는 사용자 이름입니다.")
    return render_template("signup.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route("/start_chat", methods=["POST"])
def start_chat():
    user_id = current_user.id if current_user.is_authenticated else None
    new_chat = Chat(user_id=user_id)
    db.session.add(new_chat)
    db.session.commit()
    return jsonify({"id": new_chat.id})


@app.route("/get_chat/<int:chat_id>")
def get_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id).first()
    if not chat or (chat.user_id and chat.user_id != current_user.id):
        return jsonify({"error": "Chat not found"}), 404
    messages = [{"sender": msg.sender, "text": msg.text, "timestamp": msg.timestamp} for msg in chat.messages]
    return jsonify({"id": chat.id, "messages": messages})

@app.route("/send_message", methods=["POST"])
def send_message():
    print("send_message route called")  # 추가된 로그
    data = request.json
    print("Received data:", data)  # 추가된 로그

    chat_id = data.get("chat_id")
    message_text = data.get("message")
    print(f"chat_id: {chat_id}, message_text: {message_text}")  # 추가된 로그

    chat = Chat.query.filter_by(id=chat_id).first()
    if not chat or (chat.user_id and chat.user_id != current_user.id):
        print("Chat not found")  # 추가된 로그
        return jsonify({"error": "Chat not found"}), 404

    sender = current_user.username if current_user.is_authenticated else "anonymous"
    user_message = Message(chat_id=chat_id, sender=sender, text=message_text)
    db.session.add(user_message)
    db.session.commit()

    # Get bot response
    ints = predict_class(message_text, model)
    bot_response_text = get_response(ints, intents, message_text)
    bot_message = Message(chat_id=chat_id, sender="bot", text=bot_response_text)
    db.session.add(bot_message)
    db.session.commit()

    messages = [{"sender": msg.sender, "text": msg.text, "timestamp": msg.timestamp} for msg in chat.messages]
    return jsonify({"id": chat.id, "messages": messages})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
