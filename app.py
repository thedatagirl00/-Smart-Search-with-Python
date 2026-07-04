import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

st.set_page_config(layout='wide')

st.title('Semantic Search Engine')
st.markdown('Enter a query and find semantically similar articles.')

@st.cache_resource
def load_data_and_model():
    # Our "Database" of articles
    data = [
        {'title': 'Python for Beginners', 'content': 'Learn the basics of Python programming, loops, and functions.'},
        {'title': 'Machine Learning 101', 'content': 'An introduction to neural networks and supervised learning.'},
        {'title': 'Healthy Cooking Tips', 'content': 'How to cook nutritious meals quickly using fresh ingredients.'},
        {'title': 'Best Laptops of 2026', 'content': 'Reviews of the top computing devices for work and gaming.'},
        {'title': 'The Future of AI', 'content': 'Discussing Large Language Models and the ethics of artificial intelligence.'},
        {'title': 'Yoga for Stress', 'content': 'Simple stretches and breathing exercises to relax.'}
    ]

    df = pd.DataFrame(data)

    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Combine title and content for better context
    sentences = df['title'] + ': ' + df['content']

    # Generate embeddings
    embeddings = model.encode(sentences.tolist())
    embeddings = np.array(embeddings).astype("float32")

    # Create the FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1]) 
    index.add(embeddings)
    
    return df, model, index

df, model, index = load_data_and_model()

def search(query, k=3):
    # 1. Encode the query
    query_vector = model.encode([query])
    
    # 2. Search the index
    distances, indices = index.search(np.array(query_vector).astype("float32"), k)
    
    # 3. Format results
    results = []
    for i in range(k):
        idx = indices[0][i]
        if idx != -1: 
            result = {
                'title': df.iloc[idx]['title'],
                'content': df.iloc[idx]['content'],
                'score': float(distances[0][i]) 
            }
            results.append(result)
            
    return results

query_input = st.text_input('Search for articles:', 'How to relax')

if query_input:
    st.subheader('Search Results:')
    search_results = search(query_input)
    if search_results:
        for i, res in enumerate(search_results):
            st.write(f"**{i+1}. {res['title']}** (Score: {res['score']:.4f})")
            st.write(res['content'])
            st.markdown('---')
    else:
        st.write('No results found.')
