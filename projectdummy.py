import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

class BeautyRecommender:
    def __init__(self, product_data, review_data):
        print("Initializing recommender...")
        self.product_data = product_data.copy()
        self.review_data = review_data.copy()
        
        print(f"Product data shape: {self.product_data.shape}")
        print(f"Review data shape: {self.review_data.shape}")
        
        review_columns = ['product_id', 'rating', 'skin_tone', 'skin_type', 'hair_color', 'eye_color']
        product_columns = ['product_id', 'product_name', 'brand_name', 'price_usd', 
                         'primary_category', 'secondary_category']
        
        self.review_data = self.review_data[review_columns]
        self.product_data = self.product_data[product_columns]
        
        print("\nMerging data...")
        self.data = pd.merge(
            self.review_data,
            self.product_data,
            on='product_id',
            how='inner'
        )
        print(f"Merged data shape: {self.data.shape}")
        print("Merged data columns:", self.data.columns.tolist())
        
        self.user_features = ['skin_tone', 'skin_type', 'hair_color', 'eye_color']
        self.le_dict = {}
        
        print("\nEncoding user features...")
        for feature in self.user_features:
            print(f"Processing feature: {feature}")
            self.le_dict[feature] = LabelEncoder()
            self.data[feature] = self.data[feature].fillna('unknown').astype(str).str.lower()
            self.le_dict[feature].fit(self.data[feature])
            self.data[f'{feature}_encoded'] = self.le_dict[feature].transform(self.data[feature])
        
        self.feature_matrix = self.data[[f'{feature}_encoded' for feature in self.user_features]].values
        
        self.feature_values = {
            feature: sorted(self.data[feature].unique())
            for feature in self.user_features
        }
        
    def find_similar_users(self, user_attributes, n_similar=5):
        print("\nFinding similar users...")
        print("User attributes:", user_attributes)
        
        user_encoded = []
        for feature in self.user_features:
            value = str(user_attributes.get(feature, 'unknown')).lower()
            print(f"Processing feature {feature} with value {value}")
            
            if value not in self.data[feature].unique():
                print(f"Warning: {value} not found in {feature}. Available values: {self.feature_values[feature]}")
                value = 'unknown'
            
            encoded_value = self.le_dict[feature].transform([value])[0]
            user_encoded.append(encoded_value)
        
        user_encoded = np.array(user_encoded).reshape(1, -1)
        similarities = cosine_similarity(user_encoded, self.feature_matrix)
        similar_user_indices = similarities[0].argsort()[-n_similar:][::-1]
        
        print(f"Found {len(similar_user_indices)} similar users")
        return similar_user_indices, similarities[0][similar_user_indices]
    
    def get_recommendations(self, user_attributes, n_recommendations=5, min_rating=3.5):
        print("\nGetting recommendations...")
        similar_users, similarity_scores = self.find_similar_users(user_attributes)
        
        similar_users_data = self.data.iloc[similar_users].copy()
        print(f"Found {len(similar_users_data)} reviews from similar users")
        
        if len(similar_users_data) == 0:
            print("No similar users found!")
            return pd.DataFrame()
        
        similar_users_data['similarity_score'] = similarity_scores.repeat(
            similar_users_data.groupby(similar_users_data.index).size()
        )
        
        print("Calculating product scores...")
        agg_dict = {
            'rating': ['mean', 'count'],
            'similarity_score': 'mean',
            'product_name': 'first',
            'brand_name': 'first',
            'price_usd': 'first',
            'primary_category': 'first'
        }
        
        product_scores = similar_users_data.groupby('product_id').agg(agg_dict).reset_index()
        
        product_scores.columns = ['product_id', 'avg_rating', 'review_count', 'user_similarity',
                                'product_name', 'brand_name', 'price_usd', 'primary_category']
        
        print(f"Found scores for {len(product_scores)} products")
        
        product_scores = product_scores[
            (product_scores['avg_rating'] >= min_rating) &
            (product_scores['review_count'] >= 1)
        ]
        
        if len(product_scores) == 0:
            print("No products found meeting criteria. Relaxing constraints...")
            return self.get_recommendations(user_attributes, n_recommendations, min_rating=3.0)
        
        product_scores['score'] = (
            0.4 * product_scores['avg_rating'] +
            0.4 * product_scores['user_similarity'] * 5 +
            0.2 * np.log1p(product_scores['review_count'])
        )
        
        recommendations = (product_scores
                         .sort_values('score', ascending=False)
                         .head(n_recommendations)
                         .reset_index(drop=True))
        
        recommendations['score'] = recommendations['score'].round(2)
        recommendations['avg_rating'] = recommendations['avg_rating'].round(1)
        recommendations['price_usd'] = recommendations['price_usd'].round(2)
        
        return recommendations[['product_name', 'brand_name', 'primary_category', 
                              'avg_rating', 'price_usd', 'score']]

def create_recommender(product_file, review_file):
    print(f"\nLoading data from {product_file} and {review_file}")
    try:
        product_data = pd.read_csv(product_file)
        review_data = pd.read_csv(review_file)
        recommender = BeautyRecommender(product_data, review_data)
        return recommender
    except Exception as e:
        print(f"Error creating recommender: {str(e)}")
        raise

def get_recommendations_for_user(recommender, user_attributes):
    try:
        recommendations = recommender.get_recommendations(user_attributes)
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        raise