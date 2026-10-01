import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib
import os
from groq import Groq


HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Title
st.title("Sigma Web Development RAG Bot")
st.write("Ask questions about the web development course videos.")


@st.cache_resource
def load_embeddings():
    return joblib.load("embeddings.joblib")

df = load_embeddings()

def create_embedding(text):
    # Hugging Face ki free API bge-m3 embeddings ke liye
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    
    if response.status_code != 200:
        raise Exception(f"Embedding Error: {response.text}")
        
    return response.json()[0]

def inference_llama(prompt):
    
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.2-90b-text-preview", 
        temperature=0.7
    )
    return response.choices[0].message.content

# 3. Streamlit UI Elements
incoming_query = st.text_input("Apna sawal yahan likhein...", placeholder="e.g., CSS grid kis video mein hai?")

if st.button("Jawab Dein"):
    if incoming_query:
        try:
            with st.spinner("Jawab dhoondh raha hai (Embeddings & Llama 3.2 working)..."):
                # A. Get embedding
                question_embedding = create_embedding(incoming_query)
                
                # B. Find similarity
                similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
                top_results = 5
                max_indx = similarities.argsort()[::-1][0:top_results]
                new_df = df.loc[max_indx]
                
                # C. Create Prompt
                prompt = f'''I am teaching web development in my Sigma web development course. 
                Here are video subtitle chunk containing video title, video number, start time in seconds,
                end time in seconds, the text at that time:

                {new_df[["title","number","start","end","text"]].to_json(orient="records")}
                ---------------------------------------------
                "{incoming_query}"
                User ask this question related to the video chunks, you have to answer in a human way 
                (dont mention the above format,its just for you) where and how much content is taught 
                in which video (in which video at what timestamp) and guide the user to go to that particular video. 
                If user ask unrelated question, tell him that you can only answer questions related to the course.'''

                
                answer = inference_llama(prompt)
                
               
                st.success("Answer:")
                st.write(answer)
                
        except Exception as e:
            st.error(f"System Error: {str(e)}")
    else:
        st.warning("Please enter a question!")