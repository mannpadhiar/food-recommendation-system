from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

app = FastAPI()


df = pd.read_csv("Indian_Food_Nutrition_Processed.csv")

df = df.rename(columns={
    'Dish Name': 'food_name',
    'Calories (kcal)': 'calories',
    'Carbohydrates (g)': 'carbs',
    'Protein (g)': 'protein',
    'Fats (g)': 'fat'
})

FEATURES = ['calories', 'carbs', 'protein', 'fat']


# Request schema

class Nutrition(BaseModel):
    calories: float
    carbs: float
    protein: float
    fat: float


class RequestData(BaseModel):
    target: Nutrition
    consumed: Nutrition


# Recommendation function
def get_recommendations(df, consumed, target, top_n=5):

    remaining = [
        target.calories - consumed.calories,
        target.carbs - consumed.carbs,
        target.protein - consumed.protein,
        target.fat - consumed.fat
    ]

    remaining = [max(0, x) for x in remaining]

    df_filtered = df[
        (df['calories'] <= remaining[0]) &
        (df['fat'] <= remaining[3])
    ].copy()

    if df_filtered.empty:
        return []
    
    X = df_filtered[FEATURES].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    remaining_scaled = scaler.transform([remaining])

    distances = euclidean_distances(remaining_scaled, X_scaled)

    top_indices = distances[0].argsort()[:top_n]

    results = df_filtered.iloc[top_indices][
        ['food_name', 'calories', 'carbs', 'protein', 'fat']
    ]

    return results.to_dict(orient="records")


# API endpoint
@app.post("/recommend")
def recommend(data: RequestData):

    recommendations = get_recommendations(
        df,
        data.consumed,
        data.target
    )

    print(f"remaining -------------------- {data.consumed}")
    print(f"remaining -------------------- {data.target}")

    if not recommendations:
        return {"message": "No suitable food found"}

    return {
        "recommendations": recommendations
    }