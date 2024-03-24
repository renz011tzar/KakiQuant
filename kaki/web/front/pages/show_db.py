import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from kaki.kkdatac.crypto import get_price, get_pairs
import matplotlib.pyplot as plt
from kaki.ai.ml.mod_gnn import find_similar_crypto_pairs

# Set Streamlit configuration
st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session state variables
for key in ['crypto_pair', 'analysis_result', 'graph']:
    if key not in st.session_state:
        st.session_state[key] = None

# Load cryptocurrency data
@st.cache
def load_data():
    pairs = get_pairs("kline-1D")
    data = get_price(instId=pairs, bar="1D", fields=["open", "high", "low", "close", "instId"])
    data.set_index(["instId", "timestamp"], inplace=True)
    return data

data = load_data()

# Sidebar: User Inputs
with st.sidebar:
    st.title("GNN Cryptocurrency Analysis")
    st.session_state.crypto_pair = st.selectbox("Select Cryptocurrency Pair", get_pairs("kline-1D"))
    if st.button("Load & Analyze"):
        st.session_state.analysis_result = None  # Reset analysis result

# Main Content
st.title("Cryptocurrency Analysis with GNN")

# Function to preprocess data and perform GNN analysis
def analyze_data(data, crypto_pair):
    data['timestamp'] = pd.to_datetime(data.index.get_level_values('timestamp'))
    data = data.groupby('instId').filter(lambda x: len(x) >= 60)
    data.sort_values(by=['instId', 'timestamp'], inplace=True)
    last_60_days_data = data.groupby('instId').tail(60)

    percent_change = last_60_days_data.groupby('instId')['close'].pct_change()
    last_60_days_data['Daily Return'] = percent_change
    last_60_days_data.dropna(inplace=True)

    corr_df = last_60_days_data.pivot_table(values='Daily Return', index='timestamp', columns='instId').corr()
    corr_df[corr_df < 0.7] = 0
    np.fill_diagonal(corr_df.values, 0)

    graph = nx.Graph(corr_df)
    node2vec = Node2Vec(graph, dimensions=32, p=1, q=3, walk_length=10, num_walks=600, workers=4)
    model = node2vec.fit(window=3, min_count=1, batch_words=4)

    similar_pairs = model.wv.most_similar(crypto_pair, topn=5)
    result = pd.DataFrame(similar_pairs, columns=['Ticker', 'Similarity'])
    result['Similarity'] = result['Similarity'].apply(lambda x: round(x * 100, 2))

    return result, graph

# Perform analysis and display results
if st.session_state.crypto_pair and not st.session_state.analysis_result:
    with st.spinner('Analyzing...'):
        st.session_state.analysis_result, st.session_state.graph = analyze_data(data, st.session_state.crypto_pair)
        st.success('Analysis completed!')

if st.session_state.analysis_result is not None:
    st.write(f"Similar Cryptocurrencies to {st.session_state.crypto_pair}:")
    st.table(st.session_state.analysis_result)

# Optional: Visualize the correlation graph
if st.session_state.graph is not None and st.checkbox("Show Correlation Graph"):
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(st.session_state.graph, ax=ax, with_labels=True, node_size=50, font_size=8)
    st.pyplot(fig)
