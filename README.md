# 🧠 Smart Article Recommender

A web-based article recommendation system built with Streamlit. It suggests news articles based on user interests using text clustering and similarity techniques, with interactive visualizations for deeper insights.

---

## 📊 Dataset

**UCI News Aggregator Dataset**  
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/News+Aggregator)  
- Contains ~420,000 news headlines from sources like Reuters, Huffington Post, Business Insider, etc.  
- Fields used: `TITLE`, `CATEGORY`, `URL`

---

## 🔍 Approach Summary

- **TF-IDF Vectorization**: Converts article titles into numerical vectors.
- **KMeans Clustering**: Groups similar articles into clusters.
- **Cosine Similarity**: Compares user input with articles in the closest cluster to find the most relevant.
- **Visualizations**:
  - Pie chart: Distribution of article categories
  - Bar chart: Number of articles per cluster
  - Word Cloud: Common terms in the matched cluster

---

## 📦 Dependencies

Install required packages using:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn wordcloud
