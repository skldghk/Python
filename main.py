import numpy as np
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from fuzzywuzzy import fuzz  # fuzzywuzzy를 사용하여 유사도를 계산

# 모델과 데이터 로드
lemmatizer = WordNetLemmatizer()

try:
    model = load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json', encoding='utf-8').read())
    words = json.loads(open('words.json', encoding='utf-8').read())
    classes = json.loads(open('classes.json', encoding='utf-8').read())
except Exception as e:
    print(f"Error loading model or data files: {e}")
    exit()

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
        similarity = fuzz.token_set_ratio(sentence, pattern)
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
            return random.choice(i['responses'])
    return "죄송합니다, 이해하지 못했어요. 다시 말씀해주시겠어요?"

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents, msg)
    return res

print("챗봇을 시작합니다. 종료하려면 '종료'라고 입력하세요.")
while True:
    user_input = input("You: ")
    if user_input.lower() == '종료':
        break
    response = chatbot_response(user_input)
    print(f"Bot: {response}")
