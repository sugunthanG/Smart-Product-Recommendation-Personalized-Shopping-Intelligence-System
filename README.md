# Smart Product Recommendation & Personalized Shopping Intelligence System

## Project Overview

An AI-powered recommendation engine that suggests similar products using Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity modeling.

This project simulates real-world e-commerce recommendation systems used by companies like Amazon, Flipkart, Netflix, and Spotify.

The system helps businesses improve:

- Personalized shopping experience
- Product discovery
- Customer engagement
- Product recommendation intelligence
- Conversion optimization

---

# Business Problem

Modern e-commerce platforms contain thousands of products.

Customers often struggle to:

- Discover relevant products
- Compare similar items
- Find personalized recommendations

Poor recommendation systems can lead to:

- Low customer engagement
- Reduced sales conversion
- Weak customer retention

This project solves these problems using AI-powered recommendation intelligence.

---

# Business Objective

Build an intelligent recommendation engine capable of:

- Recommending similar products
- Understanding product relationships
- Ranking products using similarity scores
- Providing explainable recommendations
- Improving personalized shopping experience

---

# Dataset Information

Dataset contains laptop product information including:

- Product Name
- Brand
- Selling Price
- Ratings
- Product Details

Dataset Size:

- 2525 rows
- 8 columns

---

# Technologies Used

## Programming Language
- Python

## Libraries
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

---

# Machine Learning Workflow

## 1. Data Preprocessing

- Removed missing values
- Cleaned product information
- Combined product features for NLP processing

## 2. NLP Feature Engineering

Used:

- TF-IDF Vectorization

to convert product text into numerical vectors.

## 3. Similarity Modeling

Used:

- Cosine Similarity

to measure product-to-product similarity.

## 4. Recommendation Engine

Generated top-N similar product recommendations.

## 5. Explainable AI Layer

Added recommendation reasoning based on similarity strength.

---

# Streamlit Dashboard Features

## Product Recommendation Dashboard

Users can:

- Search products
- Generate recommendations
- View similarity scores
- Analyze recommendation insights
- Visualize recommendation ranking

---

# Key Features

- NLP-based recommendation system
- Personalized product suggestions
- Similarity score analytics
- Explainable recommendations
- Interactive dashboard
- Real-time recommendation generation

---

# Project Structure

Smart_Product_Recommendation_System/

│

├── app/

│   └── app.py

│

├── data/

│   └── flipkart_laptops.csv

│

├── models/

│   ├── cosine_similarity.pkl

│   └── products_data.pkl

│

├── notebooks/

│   └── recommendation_system_final.ipynb

│

├── requirements.txt

├── README.md

└── .gitignore

---

# How to Run the Project

pip install -r requirements.txt

streamlit run app/app.py


# Future Improvements
- Hybrid recommendation systems
- Deep learning recommendations
- User-based collaborative filtering
- Real-time customer personalization
- Cloud deployment scalability


# Author

- Sugunthan

- Data Scientist | MERN Stack Developer | AI & ML Enthusiast

## Clone Repository

```bash
git clone <your-github-link>

