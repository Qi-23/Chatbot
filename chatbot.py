import streamlit as st

import eventregistry
from eventregistry.EventRegistry import *

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from transformers import pipeline, BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration

import torch
from scipy.spatial.distance import cosine

BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')
                                       
T5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy="False")
T5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

responses = {
    "Tell me a current news": "This is the latest news > <content>",
    "Show me the full article" : "Here is the full article > <full_content>",
    "Show me the full news" : "Sure, this is the full news > <full_content>",
        
    "What do you do?" : "",
    "What can you do?" : "",
    
    "hi": "Hello!",
    "how are you?": "I'm good, thank you!",
    "Thank you, bye" : "It is pleasure to help you, Bye!",
    "Thankyou, bye" : "It is pleasure to help you, Bye!",
    "bye": "Goodbye!",
    "quit": "Goodbye!",
    "exit": "Goodbye!",
    "Thank you" : "You are welcome!",
    "Why are you mad?" : "I am not mad.",
    "Are you happy?" : "Yes I am!"
}


YOUR_API_KEY = "5cc8f602-c9b5-4b78-9798-6f06e9325bdc"
er = EventRegistry(apiKey = YOUR_API_KEY)
location = ""
keywords = []
category = ""

result = None
full_content = None

def retrieve_news():

    uri = er.getLocationUri(location)

    q = QueryArticlesIter(
        keywords = QueryItems.OR(keywords),
        minSentiment = 0.4,
        sourceLocationUri = uri,
        categoryUri = er.getCategoryUri(category),
        dataType = ["news"])

    for each in q.execQuery(er, sortBy = "date", maxItems = 5):
        if ("advertisement" or "signin" or "sign in") not in each['body'].lower():
            return each
    
    else:
        return None 

def generate_flan_t5_response(input_text):
    input_ids = T5_tokenizer(input_text, return_tensors="pt").input_ids
    outputs = T5_model.generate(input_ids)
    response = T5_tokenizer.decode(outputs[0])
    
    if "<pad>" in response:
        response = response.replace("<pad>", "")
        
    if "</s>" in response: 
        response = response.replace("</s>", "")
        
    return response


def encode_sentence(sentence):
    tokens = BERT_tokenizer.encode(sentence, add_special_tokens=True)
    tokens_tensor = torch.tensor([tokens])
    with torch.no_grad():
        outputs = BERT_model(tokens_tensor)
        hidden_states = outputs.last_hidden_state
    sentence_embedding = torch.mean(hidden_states, dim=1).squeeze()
    return sentence_embedding.numpy()

def cosine_similarity(sentence1, sentence2):
    embedding1 = encode_sentence(sentence1)
    embedding2 = encode_sentence(sentence2)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity


def match_input(user_input, responses):
    return_value = ""
    highest_score = 0
    for key, value in responses.items():
        similarity_score = cosine_similarity(user_input, key.lower())
        if similarity_score > 0.8:
            if similarity_score > highest_score:
                highest_score = similarity_score
                return_value = value
        
    if highest_score > 0.80:
        return return_value
    return None


def find_location(user_input):
    tokens = nltk.word_tokenize(user_input)
    pos_tags = nltk.pos_tag(tokens)
    ner_tags = nltk.ne_chunk(pos_tags)
    country = None
    
    for subtree in ner_tags:
        if isinstance(subtree, nltk.Tree) and subtree.label() == "GPE":
            country_words = [word for word, tag in subtree.leaves()]
            country = " ".join(country_words)
        elif isinstance(subtree, nltk.Tree):
            if subtree[0][0].lower() == "us":
                country = "USA"
            elif subtree[0][0].lower() == "uk":
                country = "United Kingdom"
        else: 
            if subtree[0].lower() == "us":
                country = "USA"
            elif subtree[0].lower() == "uk":
                country= "United Kingdom"
                
    if country is not None and country.lower() == "united states":
        country = "USA"
    return country


def summarize(user_input):
    summarized = summarizer(user_input, max_length=250, min_length=30, do_sample=False)
    return summarized

def classify(sequence_to_classify):
    candidate_labels = ['Politic', 'Business', 'Technology','Health','Entertainment','Sports','Science','Society', 'Game', 'Fun']
    data = classifier(sequence_to_classify, candidate_labels)
    
    try:
        max_score_index = max((i, score) for i, score in enumerate(data['scores']) if score > 0.58)
        if max_score_index[1] > 0.55:
            label = data['labels'][max_score_index[0]]

        if label in ['Business', 'Health', 'Sports', 'Science', 'Society', 'Entertainment', 'Game']:
            category = label
        elif label == "Fun":
            category = "Entertainment"
        elif label == "Technology":
            keywords.append(label)
            category = "Computer"
        elif label == "Politic":
            keywords.append(label)
            keywords.append("Government")
        else:
            keywords.append(label)

    except:
        return

def chat(user_input): 
    while True:
        newChat = 1
        location = ""
        keywords = []
        category = ""

        location = find_location(user_input)
        response = match_input(user_input, responses)
           
        return response


st.title("News ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:= st.chat_input():
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chat(prompt)

    if response is None:
        response = generate_flan_t5_response(user_input)
    elif "<content>" in response:
        temp_message = f"Echo: Sure! Hold on, retrieving news ..."
        classify(user_input)
        with st.chat_message("assistant"):
            st.markdown(temp_message)
        st.session_state.messages.append({"role": "assistant", "content": temp_message})
        result = retrieve_news()

        if result is not None:
            summarized = summarize(result['body'])
            content = f"\nTitle : {result['title']}\n\nArticle : {summarized[0]['summary_text']}\n\n -------- \nURL : {result['url']}\n Date : {result['date']}\n"
            full_content = f"\nTitle : {result['title']}\n\nArticle : {result['body']}\n\n -------- \nURL : {result['url']}\n Date : {result['date']}\n"
            response = response.replace("<content>", str(content))
        else:
            content = None
            full_content = None
            response = "Required News is not found."
    elif "<full_content>" in response:
        if full_content is not None:
            response = response.replace("<full_content>", full_content)
        else:
            response = "The news is not found."

    response = f"Echo: {response}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    

