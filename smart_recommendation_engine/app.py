import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# --- Page Setup ---
st.set_page_config(page_title="Smart Article Recommender", layout="centered")

# --- Custom Styling ---
st.markdown("""
<style>
body {
    background-color: #f9f8f4;
}
.stApp {
    background-color: #f9f8f4;
    font-family: 'Segoe UI', sans-serif;
    color: #2c2c2c;
}
h1, h2, h3, h4, h5, h6, p, label {
    color: #2c2c2c !important;
}
.stTextInput > div > div > input {
    background-color: #ffffff;
    color: #000000;
    border-radius: 8px;
    padding: 0.4rem;
}
.stSlider > div {
    background-color: #ffffff;
    padding: 0.5rem;
    border-radius: 8px;
}
.title {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #333333;
    margin-bottom: 1rem;
}
.article {
    background-color: #fffaf3;
    padding: 12px;
    border-radius: 10px;
    margin-top: 10px;
    color: #2a2a2a;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
}
.article:hover {
    background-color: #fcefdc;
}
a {
    color: #007acc;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='title'><strong>Smart Article Recommender</strong></div>", unsafe_allow_html=True)
st.markdown("💡 <span style='font-size: 1.1rem;'>Get top articles based on your interest.</span>", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("uci-news-aggregator.csv")
    return df[['TITLE', 'URL', 'CATEGORY']]

df = load_data()

# --- TF-IDF + KMeans Clustering ---
@st.cache_resource
def setup_model(data, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['TITLE'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(tfidf_matrix)
    return vectorizer, kmeans, tfidf_matrix, data

vectorizer, kmeans, tfidf_matrix, df = setup_model(df)

# --- Visualizations ---
with st.expander("📊 Explore Data Insights"):
    # Pie chart of categories
    st.subheader("Article Categories Distribution")
    category_counts = df['CATEGORY'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Bar chart of cluster sizes
    st.subheader("Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax2, palette='pastel')
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Number of Articles")
    st.pyplot(fig2)

# --- User Input ---
st.markdown("### 🔍 Type a topic you're interested in:")
user_input = st.text_input("", placeholder="e.g. mental health, AI, finance, climate change")
top_n = st.slider("Number of articles to recommend", 1, 10, 5)

# --- Recommendations & WordCloud ---
if user_input:
    user_vec = vectorizer.transform([user_input])
    user_cluster = kmeans.predict(user_vec)[0]

    cluster_df = df[df['Cluster'] == user_cluster]
    cluster_tfidf = tfidf_matrix[df['Cluster'] == user_cluster]
    cluster_df['Similarity'] = cosine_similarity(user_vec, cluster_tfidf).flatten()

    top_articles = cluster_df.sort_values(by='Similarity', ascending=False).head(top_n)

    st.markdown("###  Recommended Articles:")
    for _, row in top_articles.iterrows():
        st.markdown(f"""
        <div class="article">
            <b>{row['TITLE']}</b><br>
            <a href="{row['URL']}" target="_blank">🔗 Read More</a>
        </div>
        """, unsafe_allow_html=True)

    # --- WordCloud of Cluster ---
    st.markdown("### ☁️ Word Cloud of Related Articles")
    cluster_titles = " ".join(cluster_df['TITLE'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(cluster_titles)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)
