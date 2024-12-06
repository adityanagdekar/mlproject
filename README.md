# Skincare Recommendation System

A personalized recommendation engine designed to provide tailored skincare product suggestions based on user attributes and preferences. 
This project utilizes collaborative filtering, deep learning, and user profile features to enhance customer satisfaction and business insights.

---

## **Motivation**
The beauty and skincare industry is rapidly growing, presenting customers with an overwhelming variety of products. Users often struggle to find suitable products due to unique factors like skin type, tone, and personal preferences. This recommendation system addresses these challenges by delivering personalized and relevant suggestions, improving customer satisfaction.

---

## **Aim**
The goal is to develop a recommendation engine that:
1. Leverages user profiles (e.g., skin type, tone, preferences) and product attributes.
2. Provides personalized, diverse, and accurate skincare product recommendations.
3. Enhances user experience while offering actionable insights to businesses.

---

## **Problems**
1. Customers are overwhelmed by the vast variety of available products.
2. Finding suitable skincare products is complicated by individual needs like skin type and tone.
3. Traditional recommendation systems often fail to effectively personalize suggestions.

---

## **Dataset Overview**
- **Dataset:** Sephora Dataset from Kaggle
  - Contains **8,000+ products** and **20,000+ user reviews**.
  - Includes user attributes like **skin tone**, **skin type**, **hair color**, and **eye color**.
  - Product details: price, category, ingredients, ratings, and reviews.
  - Dataset Link : (https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data)

- **Preprocessing:**
  - Missing values handled by filling with defaults (e.g., 'unknown').
  - Categorical data encoded using `LabelEncoder`.
  - Numerical features normalized using **RobustScaler** to minimize outlier effects.

---

## **Approach**
### **1. Data Analysis**
- Explore trends such as brand popularity, price distribution, and ingredient patterns.

### **2. Data Preparation**
- Encode user attributes (e.g., skin tone, type) and product attributes (e.g., price range, brand).
- Normalize numerical features for better model compatibility.

### **3. Model Development**
- **Collaborative Filtering:**
  - Uses **cosine similarity** to group users with shared preferences.
- **Deep Learning Neural Network:**
  - Combines user and product attributes to capture complex relationships.
  - Outputs personalized scores to rank product relevance.

### **4. Refinement**
- Balances accuracy, diversity, and personalization using:
  - Brand diversity techniques.
  - Weighted scoring systems.

---

## **Model Explanation**
### **Collaborative Filtering:**
- Identifies users with similar preferences using interaction data.
- Cosine similarity groups users for basic recommendations.

### **Deep Learning-Based Neural Network:**
- **Input:** User attributes (e.g., skin tone) and product attributes (e.g., price, category).
- **Architecture:** Fully connected layers to learn complex relationships.
- **Output:** Personalized product ranking scores.

---

## **Design Decisions**
1. **Neural Network:**
   - Chosen for capturing non-linear relationships over traditional methods like cosine similarity.
2. **Weighted Scoring:**
   - Balances factors like ratings, prices, and brand diversity for fairness and accuracy.
3. **Preprocessing Choices:**
   - Categorical features encoded for better embedding representations.
   - RobustScaler applied to numerical features to mitigate outliers.

---

## **Model Comparison**
### **Baseline Approach:**
- Used cosine similarity for lightweight user-product matching.

### **Advanced Model:**
- Deep learning significantly improved recommendation diversity and personalization.

### **Performance Metrics:**
- Evaluated using **Precision**, **Recall**, and **F1 Score**.
- The advanced model outperformed the baseline across all metrics.

---

## **Results**
- The deep learning model demonstrated superior performance compared to the baseline, providing diverse and personalized recommendations.
- Improved metrics validate the advanced model's ability to address key challenges.

---

## **Challenges and Improvements**
### **Challenges:**
1. Sparse user-product interactions in the dataset.
2. Balancing recommendation diversity and accuracy.

### **Future Directions:**
1. Experiment with advanced architectures like transformers.
2. Incorporate dynamic feedback loops to improve real-time recommendations.

---
