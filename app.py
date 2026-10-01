import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib
import os
import google.generativeai as genai


HF_TOKEN = st.secrets["HF_TOKEN"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


@st.cache_resource
def load_data():
    return joblib.load("embeddings.joblib")

df = load_data()
genai.configure(api_key=GEMINI_API_KEY)

def create_embedding(text):
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    if response.status_code != 200:
        st.error(f"Embedding Error: {response.text}")
        return None
    return response.json()[0]

def inference_gemini(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- Streamlit UI ---
st.title("web development course Q&A")
st.write("Ask questions about the web development course videos.")

user_query = st.text_input("Ask your question here:")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Processing your question..."):
            question_embedding = create_embedding(user_query)
            
            if question_embedding:
                similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
                max_indx = similarities.argsort()[::-1][0:5]
                new_df = df.loc[max_indx]
                
                prompt = f'''I am teaching web development in my Sigma web development course. 
                Here are video subtitle chunk containing video title, video number, start time in seconds,
                end time in seconds, the text at that time:

                {new_df[["title","number","start","end","text"]].to_json(orient="records")}
                ---------------------------------------------
                "{user_query}"
                User ask this question related to the video chunks, you have to answer in a human way 
                (dont mention the above format,its just for you) where and how much content is taught 
                in which video (in which video at what timestamp) and guide the user to go to that particular video. 
                If user ask unrelated question, tell him that you can only answer questions related to the course.'''

                answer = inference_gemini(prompt)
                st.success("Answer")
                st.write(answer)
    else:
        st.warning("Please Ask a question before clicking the button.")