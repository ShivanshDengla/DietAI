import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'template'))
app.template_folder = template_dir

class Recommender:
    
    def __init__(self):
        self.df = pd.read_csv('dataset.csv')
    
    def get_features(self):
        nutrient_dummies = self.df.Nutrient.str.get_dummies()
        disease_dummies = self.df.Disease.str.get_dummies(sep=' ')
        diet_dummies = self.df.Diet.str.get_dummies(sep=' ')
        feature_df = pd.concat([nutrient_dummies, disease_dummies, diet_dummies], axis=1)
        return feature_df
    
    def k_neighbor(self, inputs):
        feature_df = self.get_features()
        model = NearestNeighbors(n_neighbors=40, algorithm='ball_tree')
        model.fit(feature_df)
        df_results = pd.DataFrame(columns=list(self.df.columns))
        distances, indices = model.kneighbors(inputs)
        for i in list(indices):
            df_results = pd.concat([df_results, self.df.loc[i]])  # Use concat instead of append
        df_results = df_results.filter(['Name', 'Nutrient', 'Veg_Non', 'Price', 'Review', 'Diet', 'Disease', 'description'])
        df_results = df_results.drop_duplicates(subset=['Name'])
        df_results = df_results.reset_index(drop=True)
        return df_results


ob = Recommender()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == "POST":
        # Collect user input data from the form
        categories = {
            'calcium': 0, 'carbohydrates': 0, 'chloride': 0, 'fiber': 0, 'iodine': 0, 'iron': 0, 'magnesium': 0,
            'manganese': 0, 'phosphorus': 0, 'potassium': 0, 'protien': 0, 'selenium': 0, 'sodium': 0, 'vitamin_a': 0,
            'vitamin_c': 0, 'vitamin_d': 0, 'vitamin_e': 0, 'anemia': 0, 'cancer': 0, 'diabetes': 0, 'eye_disease': 0,
            'goitre': 0, 'heart_disease': 0, 'hypertension': 0, 'kidney_disease': 0, 'obesity': 0, 'pregnancy': 0,
            'rickets': 0, 'scurvy': 0, 'Mediterranean_diet': 0, 'alkaline_diet': 0, 'dash_diet': 0,
            'gluten_free_diet': 0, 'high_fiber_diet': 0, 'high_protein_diet': 0, 'hormone_diet': 0,
            'ketogenic_diet': 0, 'low_carb_diet': 0, 'low_fat_diet': 0, 'low_sodium_diet': 0, 'omni_diet': 0,
            'paleo_diet': 0, 'type_a_diet': 0, 'type_o_diet': 0, 'vegan_diet': 0
        }

        for category in categories:
            if request.form.get(category):
                categories[category] = 1

        final_input = list(categories.values())
        results = ob.k_neighbor([final_input])
        
        return render_template('result.html', results=results.to_dict('records'))
    
    return render_template('template/index2.html')

if __name__ == '__main__':
    app.run(debug=True)