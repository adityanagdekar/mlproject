from projectdummy import create_recommender, get_recommendations_for_user
import pandas as pd

def main():
    print("Starting recommendation process...")
    
    try:
    
        recommender = create_recommender('product_info.csv', 'reviews_500-750.csv')
        
        skin_tone  = input("Print enter the skin tone: ")
        skin_type =  input("Print skin type: ")
        hair_color = input("Enter hair color: ")
        eye_color = input("Enter eye color: ")
        user_attributes = {
            'skin_tone': skin_tone,
            'skin_type': skin_type,
            'hair_color': hair_color,
            'eye_color': eye_color
        }
        
        print("\nGetting recommendations for user:", user_attributes)
        recommendations = get_recommendations_for_user(recommender, user_attributes)
        
        if len(recommendations) > 0:
            print("\nRecommended products:")
            print(recommendations)
        else:
            print("\nNo recommendations found!")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main()