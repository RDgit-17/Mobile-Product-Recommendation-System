import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "D:/Project 4 - Global Mobile Reviews/Mobile Recommendation/encoded_data.csv"

features = ["Price_usd", "Rating", "Battery_life_rating", "Camera_rating",
            "Performance_rating", "Design_rating", "Display_rating"]

names = ["Budget", "Mid-range", "Upper mid-range", "Premium"]

df = pd.read_csv(DATA_PATH)

# Product table (one row for each phone model)
products = df.groupby(["Brand", "Model"])[features].mean().reset_index()
products["Reviews"] = df.groupby(["Brand", "Model"]).size().values

# Scaling (price on its real range, ratings on the 1 to 5 scale)
price_min = df["Price_usd"].min()
price_max = df["Price_usd"].max()


def scale_features(frame):
    scaled = frame[features].copy()
    scaled["Price_usd"] = (scaled["Price_usd"] - price_min) / (price_max - price_min)
    for col in features[1:]:
        scaled[col] = (scaled[col] - 1) / 4
    return scaled


x = scale_features(products)

# K-Means with 4 clusters, named from cheapest to most expensive
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
products["Cluster"] = kmeans.fit_predict(x)

cluster_prices = products.groupby("Cluster")["Price_usd"].mean().sort_values()

segment_names = {}
i = 0
for cluster in cluster_prices.index:
    segment_names[cluster] = names[i]
    i = i + 1

products["Segment"] = products["Cluster"].map(segment_names)

# Similarity of every phone with every other phone
similarity = cosine_similarity(x)

# Currency of each country
country_info = df.groupby("Country")[["Currency", "Exchange_rate_to_usd"]].first()


def get_currency(country):
    currency = country_info.loc[country, "Currency"]
    rate = country_info.loc[country, "Exchange_rate_to_usd"]
    return currency, rate


def add_local_price(result, country):
    currency, rate = get_currency(country)
    result["Price_" + currency] = (result["Price_usd"] * rate).round(2)
    return result


def tidy(result, top_n):
    result = result.sort_values("Similarity", ascending=False).head(top_n)
    result["Price_usd"] = result["Price_usd"].round(2)
    result["Rating"] = result["Rating"].round(2)
    result["Similarity"] = result["Similarity"].round(4)
    return result.reset_index(drop=True)


def make_new_phone(price, rating, battery, camera, performance, design, display):
    new_phone = pd.DataFrame({"Price_usd": [price],
                              "Rating": [rating],
                              "Battery_life_rating": [battery],
                              "Camera_rating": [camera],
                              "Performance_rating": [performance],
                              "Design_rating": [design],
                              "Display_rating": [display]})
    return scale_features(new_phone)


def predict_segment(price, rating, battery, camera, performance, design, display):
    new_phone = make_new_phone(price, rating, battery, camera, performance, design, display)
    cluster = kmeans.predict(new_phone)[0]
    return segment_names[cluster]


def recommend_for_new_phone(price, rating, battery, camera, performance, design, display, top_n=5):
    new_phone = make_new_phone(price, rating, battery, camera, performance, design, display)

    result = products[["Brand", "Model", "Segment", "Price_usd", "Rating"]].copy()
    result["Similarity"] = cosine_similarity(new_phone, x)[0]
    return tidy(result, top_n)


def recommend_similar(model_name, top_n=5, same_segment_only=False):
    row = products[products["Model"] == model_name].index[0]

    result = products[["Brand", "Model", "Segment", "Price_usd", "Rating"]].copy()
    result["Similarity"] = similarity[row]
    result = result.drop(row)

    if same_segment_only:
        result = result[result["Segment"] == products.loc[row, "Segment"]]

    return tidy(result, top_n)


def recommend_by_budget(budget, min_rating=1.0, priority="Rating", brand="All", top_n=5):
    result = products[products["Price_usd"] <= budget]
    result = result[result["Rating"] >= min_rating]

    if brand != "All":
        result = result[result["Brand"] == brand]

    result = result.sort_values(priority, ascending=False)

    columns = ["Brand", "Model", "Segment", "Price_usd", "Rating"]
    if priority != "Rating":
        columns.append(priority)

    return result[columns].head(top_n).round(2).reset_index(drop=True)
