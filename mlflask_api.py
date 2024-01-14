import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route('/recommend')

def recommend():
    df_destinations = pd.read_csv('Tours Data.csv')

    df_user_profiles = pd.read_csv('Tourist History.csv', dtype={'Duration': int, 'Budget': int})

    label_encoders = {}
    categorical_features = ['Tour Type', 'Tour Category']
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        df_destinations[feature] = label_encoders[feature].fit_transform(df_destinations[feature])


    accumulated_recommendations = []

    user_name = 'Rehan Gondal'

    user_profiles = df_user_profiles[df_user_profiles['Tourist Name'] == user_name]

    if user_profiles.empty:
        print(f"No user profiles found for {user_name}.")
    else:
    
        k = 3  
        features = ['Tour Type', 'Tour Category', 'Duration', 'Budget']  
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(df_destinations[features])  


    #print(user_profiles)
        aggregated_user_profile = user_profiles.mode().loc[0]
    #print(aggregated_user_profile)

        for feature in categorical_features:
            aggregated_user_profile[feature] = label_encoders[feature].transform([aggregated_user_profile[feature]])[0]

        encoded_user_profile = aggregated_user_profile[features].values
    #print(encoded_user_profile)
        user_profile_array = np.array(encoded_user_profile).reshape(1, -1)
    #print(user_profile_array)
    
        distances, indices = knn.kneighbors(user_profile_array)
    #print(distances,indices)

        recommended_destinations = df_destinations.iloc[indices[0]]['Destination point'].tolist()


        accumulated_recommendations.extend(recommended_destinations)


        unique_accumulated_recommendations = list(set(accumulated_recommendations))

        if len(unique_accumulated_recommendations) == 0:
            print(f"No recommendations found for {user_name}.")
        else:
            print(f"Recommended Destinations for {user_name}:")
            print(unique_accumulated_recommendations)
            return jsonify(unique_accumulated_recommendations),200
            
            
if __name__ == '__main__':
    app.run(debug=True)

