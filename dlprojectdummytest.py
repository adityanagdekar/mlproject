import pandas as pd
import numpy as np
from dlprojectdummy import DeepLearningRecommender

def load_and_prepare_data(product_file='product_info.csv', 
                         review_file='reviews_250-500.csv'):
    try:
        products = pd.read_csv(product_file)
        reviews = pd.read_csv(review_file)
        print(f"Loaded {len(products)} products and {len(reviews)} reviews")
        data = pd.merge(reviews, products, on='product_id', how='inner')
        print(f"Merged data shape: {data.columns}")  
        cleaned_data = pd.DataFrame({
            'skin_tone': data['skin_tone'],
            'skin_type': data['skin_type'],
            'hair_color': data['hair_color'],
            'eye_color': data['eye_color'],
            'brand_name': data['brand_name_x'],
            'primary_category': data['primary_category'], 
            'secondary_category': data['secondary_category'],
            'price_usd': data['price_usd_x'],
            'rating': data['rating_x'],
            'is_recommended': data['is_recommended'],
            'primary_category':data['primary_category']
        })
        cleaned_data['primary_category'] = cleaned_data['primary_category'].fillna('unknown')
        cleaned_data['secondary_category'] = cleaned_data['secondary_category'].fillna('unknown')
        
        cleaned_data = cleaned_data.dropna(subset=['is_recommended'])
        
        print("\nTarget value counts ")
        print(cleaned_data['is_recommended'].value_counts(dropna=False))
      
        for col in ['skin_tone', 'skin_type', 'hair_color', 'eye_color', 'brand_name']:
            cleaned_data[col] = cleaned_data[col].fillna('unknown')
     
        for col in ['price_usd', 'rating']:
            median_val = cleaned_data[col].median()
            cleaned_data[col] = cleaned_data[col].fillna(median_val)

        cleaned_data['is_recommended'] = cleaned_data['is_recommended'].astype(int)

        print("\nFinal missing values:")
        print(cleaned_data.isnull().sum())
        
        features = cleaned_data.drop('is_recommended', axis=1)
        target = cleaned_data['is_recommended']
        
        print("\nFinal shapes:")
        print(f"Features: {features.shape}")
        print(f"Target: {target.shape}")
        print("\nFinal target distribution:")
        print(target.value_counts(normalize=True))
        
        return features, target

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        print("\nDetailed error information:")
        print(traceback.format_exc())
        raise

def main():
    print("Starting Recommender System Training...")
    try:
        data, target = load_and_prepare_data()
        
        if data is None or target is None:
            raise ValueError("Failed to load data")
 
        print("Features info:")
        print(data.info())
        print("\nFeatures description:")
        print(data.describe())
        print("\nTarget distribution:")
        print(target.value_counts(normalize=True))
    
        print("\nInitializing recommender system...")
        recommender = DeepLearningRecommender(
            user_features=['skin_tone', 'skin_type', 'hair_color', 'eye_color'],
            categorical_features=['brand_name', 'primary_category', 'secondary_category'],
            product_features=['price_usd', 'rating']
        )

        print("\nTraining model...")
        model, history = recommender.train(
            data=data,
            target=target,
            epochs=50,
            batch_size=32,
            early_stopping_patience=10
        )

        model.save('recommender_model.keras')
        print("\nModel saved as 'recommender_model.keras'")
        
        return recommender, model, history
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nDetailed error information:")
        print(traceback.format_exc())
        return None, None, None

if __name__ == "__main__":
    recommender, model, history = main()