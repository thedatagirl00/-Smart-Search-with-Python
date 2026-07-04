### README.md for GitHub

```markdown
# Semantic Search Engine

## Project Overview

This project implements a lightweight semantic search engine that goes beyond keyword matching, understanding the *intent* and *meaning* behind your queries. It leverages state-of-the-art natural language processing (NLP) models from `sentence-transformers` to convert text into numerical vector embeddings and uses Facebook AI Similarity Search (FAISS) for efficient similarity comparisons. The entire application is packaged and deployed as an interactive web application using Streamlit.

## How it Works

1.  **Data Ingestion**: A collection of articles (or documents) is loaded.
2.  **Embedding Generation**: Each article's title and content are combined and transformed into high-dimensional numerical vectors (embeddings) using a pre-trained `SentenceTransformer` model (`all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the text.
3.  **FAISS Indexing**: The generated embeddings are indexed using FAISS, a library for efficient similarity search. This allows for rapid lookup of the most similar vectors.
4.  **Semantic Search**: When a user enters a query, it's also converted into an embedding. FAISS then quickly finds the closest article embeddings in the index, returning semantically relevant results.
5.  **Streamlit Deployment**: The entire functionality is wrapped in a Streamlit application, providing an intuitive user interface for searching and displaying results.

## Features

*   **Intent-Based Search**: Finds relevant articles even if exact keywords aren't present.
*   **Fast Similarity Search**: Utilizes FAISS for efficient nearest neighbor lookups.
*   **Interactive UI**: User-friendly web interface built with Streamlit.
*   **Scalable**: The core components (Sentence Transformers, FAISS) are designed for performance.

## Technologies Used

*   **Python**
*   **pandas**: For data handling.
*   **sentence-transformers**: For generating text embeddings.
*   **numpy**: For numerical operations, especially with embeddings.
*   **faiss-cpu**: For efficient similarity search in high-dimensional spaces.
*   **Streamlit**: For creating the interactive web application.

## Setup and Local Run

To run this project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd semantic-search-engine
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    The required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    
    **`requirements.txt` content:**
    ```
    streamlit
    pandas
    sentence-transformers
    numpy
    faiss-cpu
    ```

4.  **Run the Streamlit application**:
    The main application logic is in `app.py`.
    ```bash
    streamlit run app.py
    ```
    
    **`app.py` content:**
    ```python
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
    ```

    This will open the Streamlit app in your default web browser.

## Deployment on Streamlit Cloud

This application is designed to be easily deployable on [Streamlit Cloud](https://share.streamlit.io/).

1.  **Push to GitHub**: Ensure your `app.py` and `requirements.txt` files are in the root directory of a GitHub repository.
2.  **Connect Streamlit Cloud**: Go to [Streamlit Cloud](https://share.streamlit.io/), log in, and select 'New app'. Connect your GitHub repository.
3.  **Deploy**: Choose your repository, ensure `app.py` is the main file, and click 'Deploy!'. Streamlit Cloud will handle the environment setup and provide a public URL for your app.

```
