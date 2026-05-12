import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np


# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="Smart Product Recommendation System",
    page_icon="🛒",
    layout="wide"
)


# ===================================================
# PATH CONFIGURATION
# ===================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "products_data.pkl"
)

SIMILARITY_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "cosine_similarity.pkl"
)


# ===================================================
# LOAD FILES
# ===================================================
df = joblib.load(DATA_PATH)

cosine_sim = joblib.load(SIMILARITY_PATH)


# ===================================================
# FIX COSINE SIMILARITY FORMAT
# ===================================================

# Convert dataframe to numpy array if needed
if isinstance(cosine_sim, pd.DataFrame):
    cosine_sim = cosine_sim.values

# Force numeric datatype
cosine_sim = np.array(cosine_sim, dtype=float)


# ===================================================
# TITLE
# ===================================================
st.title(
    "Smart Product Recommendation & Personalized Shopping Intelligence"
)

st.markdown("""
This AI-powered recommendation engine recommends similar products using:

- NLP (Natural Language Processing)
- TF-IDF Vectorization
- Cosine Similarity
- Intelligent Ranking Algorithms

### Business Benefits
- Improve customer engagement
- Increase product discovery
- Enhance personalized shopping
- Support intelligent product recommendations
""")


# ===================================================
# RECOMMENDATION FUNCTION
# ===================================================
def recommend_products(product_name, top_n=5):

    # Find matching product
    matches = df[df["Name"] == product_name]

    if matches.empty:
        return None

    # Get index
    idx = matches.index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    # Convert scores to float safely
    clean_scores = []

    for item in similarity_scores:

        try:
            clean_scores.append(
                (item[0], float(item[1]))
            )

        except:
            continue

    # Sort descending
    similarity_scores = sorted(
        clean_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove self recommendation
    similarity_scores = similarity_scores[1:top_n + 1]

    # Product indices
    product_indices = [
        i[0] for i in similarity_scores
    ]

    # Recommendation dataframe
    recommendations = df.iloc[product_indices][[
        "Name",
        "Brand",
        "Selling Price",
        "Ratings"
    ]].copy()

    # Similarity %
    recommendations["Similarity Score"] = [
        round(score[1] * 100, 2)
        for score in similarity_scores
    ]

    # Explainability
    explanations = []

    for score in recommendations["Similarity Score"]:

        if score >= 85:
            explanations.append(
                "Highly similar product with nearly identical specifications"
            )

        elif score >= 70:
            explanations.append(
                "Strong recommendation based on product features and category similarity"
            )

        elif score >= 50:
            explanations.append(
                "Moderately related recommendation"
            )

        else:
            explanations.append(
                "Alternative recommendation with lower similarity"
            )

    recommendations["Recommendation Reason"] = explanations

    return recommendations


# ===================================================
# SIDEBAR
# ===================================================
st.sidebar.header("Recommendation Settings")

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)


# ===================================================
# PRODUCT SEARCH
# ===================================================
st.subheader("Product Search")

product_list = sorted(
    df["Name"].dropna().unique()
)

selected_product = st.selectbox(
    "Select a Product",
    product_list
)


# ===================================================
# BUTTON
# ===================================================
if st.button("Generate Recommendations"):

    recommendations = recommend_products(
        selected_product,
        top_n
    )

    if recommendations is None:

        st.error("Product not found.")

    else:

        # ===================================================
        # RESULTS
        # ===================================================
        st.markdown("---")

        st.subheader("Recommended Products")

        st.dataframe(
            recommendations,
            use_container_width=True
        )


        # ===================================================
        # ANALYTICS
        # ===================================================
        st.markdown("---")

        st.subheader("Recommendation Analytics")

        highest_score = recommendations[
            "Similarity Score"
        ].max()

        avg_rating = recommendations[
            "Ratings"
        ].mean()

        avg_price = recommendations[
            "Selling Price"
        ].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Highest Similarity",
                f"{highest_score:.2f}%"
            )

        with col2:
            st.metric(
                "Average Rating",
                f"{avg_rating:.2f}"
            )

        with col3:
            st.metric(
                "Average Price",
                f"₹{avg_price:,.0f}"
            )


        # ===================================================
        # VISUALIZATION
        # ===================================================
        st.markdown("---")

        st.subheader("Similarity Score Visualization")

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.barh(
            recommendations["Name"],
            recommendations["Similarity Score"]
        )

        ax.set_xlabel("Similarity Score (%)")

        ax.set_ylabel("Recommended Products")

        ax.set_title("Top Recommended Products")

        plt.xticks(rotation=0)

        st.pyplot(fig)


        # ===================================================
        # BUSINESS INSIGHTS
        # ===================================================
        st.markdown("---")

        st.subheader("Recommendation Intelligence")

        st.info(f"""
### AI Recommendation Summary

The recommendation engine analyzed:

- Product specifications
- Brand similarity
- Product descriptions
- Category-level patterns
- NLP-based feature similarity

### Key Insights

- Highest similarity score: {highest_score:.2f}%
- Average product rating: {avg_rating:.2f}
- Average product price: ₹{avg_price:,.0f}

### Recommendation Strategy

Products are recommended using:

- TF-IDF vectorization
- Cosine similarity modeling
- NLP text analysis
- Intelligent ranking systems

This simulates real-world e-commerce recommendation engines.
""")


        # ===================================================
        # TOP RECOMMENDATION
        # ===================================================
        st.markdown("---")

        st.subheader("Best Recommendation")

        best_product = recommendations.iloc[0]

        st.success(f"""
Product:
{best_product['Name']}

Brand:
{best_product['Brand']}

Price:
₹{best_product['Selling Price']}

Similarity Score:
{best_product['Similarity Score']}%

Reason:
{best_product['Recommendation Reason']}
""")


# ===================================================
# FOOTER
# ===================================================
st.markdown("---")

st.caption(
    "Built with Machine Learning, NLP, TF-IDF, Cosine Similarity, Pandas, Scikit-learn, and Streamlit"
)