import pandas as pd
import numpy as np
from dlprojectdummy import DeepLearningRecommender
from tensorflow.keras.models import load_model
import tensorflow as tf

def prepare_batch_data(user_features, products, brand_to_idx):
    n_products = len(products)
    
    user_data = pd.DataFrame([user_features] * n_products)
    
    products_copy = products.copy()
    
    products_copy['brand_idx'] = products_copy['brand_name'].map(
        lambda x: brand_to_idx[x] % 65  
    )
    
    products_copy['price_category'] = pd.cut(
        products_copy['price_usd'],
        bins=[0, 25, 75, float('inf')],
        labels=['budget', 'mid_range', 'premium']
    )
    
    batch_data = pd.DataFrame({
        'skin_tone': user_data['skin_tone'],
        'skin_type': user_data['skin_type'],
        'hair_color': user_data['hair_color'],
        'eye_color': user_data['eye_color'],
        'brand_name': products_copy['brand_idx'], 
        'price_usd': products_copy['price_usd'],
        'rating': products_copy['rating'],
        'price_category': products_copy['price_category']
    })
    
    print("\nBrand mapping statistics:")
    print(f"Number of unique brands after mapping: {len(batch_data['brand_name'].unique())}")
    print(f"Brand index range: {batch_data['brand_name'].min()} to {batch_data['brand_name'].max()}")
    
    return batch_data

def load_product_catalog(file_path='product_info.csv'):
    try:
        products = pd.read_csv(file_path)
        products = products.sample(frac=1, random_state=None)

        print(f"\nLoaded {len(products)} products from catalog")
        
        unique_brands = sorted(products['brand_name'].unique()) 
        brand_to_idx = {brand: idx for idx, brand in enumerate(unique_brands)}
        
        print("\nProduct catalog summary:")
        print(f"Number of unique product names: {products['product_name'].nunique()}")
        print(f"Number of unique brands: {len(unique_brands)}")
        
        required_columns = ['product_name', 'brand_name', 'price_usd', 'rating']
        if not all(col in products.columns for col in required_columns):
            raise ValueError(f"Product catalog missing required columns. Required: {required_columns}")
            
        return products, brand_to_idx
    except Exception as e:
        print(f"Error loading product catalog: {str(e)}")
        raise

def get_user_input(review_file='reviews_250-500.csv'):
    print("\nPlease answer few questions:")

    reviews = pd.read_csv(review_file)
    
    valid_options = {
        'skin_tone': sorted([str(x) for x in reviews['skin_tone'].dropna().unique()]),
        'skin_type': sorted([str(x) for x in reviews['skin_type'].dropna().unique()]),
        'hair_color': sorted([str(x) for x in reviews['hair_color'].dropna().unique()]),
        'eye_color': sorted([str(x) for x in reviews['eye_color'].dropna().unique()])
    }
    
    
    user_features = {}
    for feature, options in valid_options.items():
        while True:
            print(f"\nValid options for {feature}: {', '.join(options)}")
            value = input(f"Enter your {feature}: ").lower().strip()
            if value in options:
                user_features[feature] = value
                break
            else:
                print(f"Invalid input. Please choose from: {', '.join(options)}")
    
    return user_features



def normalize_score(series, inverse=False):
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        return pd.Series(1.0, index=series.index)
    
    score = (series - min_val) / (max_val - min_val)
    return 1 - score if inverse else score

def generate_explanation(row, avg_price):
    factors = []
    if row['rating'] >= 4.0:
        factors.append(f"high rating ({row['rating']:.1f}/5)")
    if row['price_usd'] <= 50:
        factors.append(f"competitive price (${row['price_usd']:.2f})")
    
    price_diff = ((avg_price - row['price_usd']) / avg_price) * 100
    if abs(price_diff) > 10:
        if price_diff > 0:
            factors.append(f"{abs(price_diff):.0f}% below average price")
        else:
            factors.append(f"{abs(price_diff):.0f}% above average price")
    
    base = f"Recommended due to: {', '.join(factors)}" if factors else "Based on overall features"
    confidence = row['final_score'] * 100
    
    return f"{base} (Confidence: {confidence:.1f}%)"



def print_diversity_metrics(recommendations):
    print("\nRecommendation Diversity:")
    print(f"Unique brands: {len(recommendations['brand_name'].unique())}")
    print(f"Price range: ${recommendations['price_usd'].min():.2f} - ${recommendations['price_usd'].max():.2f}")
    print(f"Rating range: {recommendations['rating'].min():.1f} - {recommendations['rating'].max():.1f}")


def make_predictions(recommender, model, user_features, products, brand_to_idx, top_k=5):
    try:
        print(f"\nPreparing to analyze {len(products)} products...")
        
        batch_data = prepare_batch_data(user_features, products, brand_to_idx)
        
        processed_data = recommender.preprocess_data(batch_data)
        test_inputs = recommender.prepare_model_inputs(processed_data)
        
        with tf.device('/CPU:0'):
            base_predictions = model.predict(test_inputs, verbose=0)
        
        results = pd.DataFrame({
            'product_name': products['product_name'],
            'brand_name': products['brand_name'],
            'price_usd': products['price_usd'],
            'rating': products['rating'],
            'primary_category': products['primary_category'],
            'base_score': base_predictions.flatten()
        })
        
        results['base_score'] = results['base_score'].rank(pct=True)
        
        price_ranges = {
            'budget': (0, 25),
            'affordable': (25, 50),
            'mid_range': (50, 100),
            'premium': (100, float('inf'))
        }
        
        median_price = results['price_usd'].median()
        results['price_score'] = results['price_usd'].apply(
            lambda x: 1 - (abs(x - median_price) / median_price)
        ).clip(0, 1)
        
        results['rating_score'] = results['rating'].rank(pct=True)
        
        brand_counts = results['brand_name'].value_counts()
        brand_scores = 1 / np.log1p(brand_counts)
        results['brand_diversity'] = results['brand_name'].map(brand_scores)
        
        avg_price = results['price_usd'].mean()
        price_weight = 0.3 if avg_price > 50 else 0.2
        
        weights = {
            'base_score': 0.4,
            'rating_score': 0.25,
            'price_score': price_weight,
            'brand_diversity': 0.35 - price_weight  
        }
        
        np.random.seed(None)  
        results['random_factor'] = np.random.uniform(0.95, 1.05, size=len(results))
        
        results['final_score'] = (
            weights['base_score'] * results['base_score'] +
            weights['rating_score'] * results['rating_score'] +
            weights['price_score'] * results['price_score'] +
            weights['brand_diversity'] * results['brand_diversity']
        ) * results['random_factor']
        
        results['recommendation_strength'] = pd.cut(
            results['final_score'],
            bins=[-np.inf, 0.3, 0.7, np.inf],
            labels=['Weak', 'Medium', 'Strong']
        )
        
        results['explanation'] = results.apply(
            lambda row: generate_explanation(row, results['price_usd'].mean()),
            axis=1
        )
        
        recommendations = select_diverse_recommendations_enhanced(results, top_k)
        
        return recommendations[['product_name', 'brand_name', 'price_usd', 
                              'rating', 'recommendation_strength', 'explanation',
                              'final_score','primary_category']]
        
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        raise

def select_diverse_recommendations_enhanced(results, top_k):
    top_recommendations = []
    seen_brands = set() 
    
    top_100 = results.nlargest(100, 'final_score')
    
    while len(top_recommendations) < top_k and not top_100.empty:
        probabilities = top_100['final_score'] / top_100['final_score'].sum()
        selected_idx = np.random.choice(top_100.index, p=probabilities)
        product = top_100.loc[selected_idx]
        
        if product['brand_name'] not in seen_brands or len(seen_brands) >= top_k/2:
            top_recommendations.append(product)
            seen_brands.add(product['brand_name'])
        
        top_100 = top_100.drop(selected_idx)
    
    return pd.DataFrame(top_recommendations)

def main():
    print("Starting recommendation process...")
    try:
        print("\nInitializing recommender system...")
        recommender = DeepLearningRecommender(
            user_features=['skin_tone', 'skin_type', 'hair_color', 'eye_color'],
            categorical_features=['brand_name'],
            product_features=['price_usd', 'rating']
        )
        
        print("Loading trained model...")
        try:
            model = load_model('recommender_model.keras')
            recommender.model = model
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please ensure you have trained the model first using dlprojectdummytest.py")
            return
        
        user_features = get_user_input()
        print("\nUser profile:", user_features)
        
        products, brand_to_idx = load_product_catalog()
        
        print("\nGenerating personalized recommendations...")
        recommendations = make_predictions(
            recommender, model, user_features, products, brand_to_idx=brand_to_idx)
        
        print("\nTop 5 Recommended Products:")
        display_cols = [
            'product_name', 'brand_name', 'price_usd', 
            'rating', 'recommendation_strength', 'explanation','primary_category'
        ]
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(recommendations[display_cols].to_string(index=False))
        
        print("\nRecommendation Statistics:")
        if not recommendations.empty:
            print(f"Average recommendation score: {recommendations['final_score'].mean():.3f}")
            print("\nRecommendation strength distribution:")
            print(recommendations['recommendation_strength'].value_counts())
        else:
            print("No recommendations generated.")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print("\nDetailed error information:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()