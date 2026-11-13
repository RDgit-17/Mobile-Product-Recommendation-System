import streamlit as st
import plotly.express as px
from pipeline import (df, products, predict_segment, recommend_for_new_phone, recommend_similar,recommend_by_budget, get_currency, add_local_price)

SEGMENT_NAMES = ["Budget", "Mid-range", "Upper mid-range", "Premium"]

SEGMENT_COLORS = {"Budget": "green", "Mid-range": "gold",
                  "Upper mid-range": "orange", "Premium": "red"}

st.set_page_config(page_title="Mobile Segmentation and Recommendation", layout="wide")

# Country and local currency
countries = sorted(df["Country"].unique())
country = st.sidebar.selectbox("Your country", countries, index=countries.index("USA"))
currency, rate = get_currency(country)
st.sidebar.write("1 USD = " + str(round(rate, 2)) + " " + currency)


# ---------------------------------------------------------
#                       Dashboard tab
# ---------------------------------------------------------
def show_overview():
    st.header("Data Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Reviews", len(df))
    col2.metric("Brands", df["Brand"].nunique())
    col3.metric("Phone models", df["Model"].nunique())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Countries", df["Country"].nunique())
    col2.metric("Average price (USD)", round(df["Price_usd"].mean(), 2))
    col3.metric("Average price (" + currency + ")", round(df["Price_usd"].mean() * rate, 2))
    col4.metric("Average rating", round(df["Rating"].mean(), 2))

    col1, col2 = st.columns(2)

    brand_counts = df["Brand"].value_counts().reset_index()
    brand_counts.columns = ["Brand", "Reviews"]
    fig = px.bar(brand_counts, x="Brand", y="Reviews", title="Reviews per brand")
    fig.update_yaxes(range=[6700, 6900])
    col1.plotly_chart(fig)

    country_counts = df["Country"].value_counts().reset_index()
    country_counts.columns = ["Country", "Reviews"]
    fig = px.bar(country_counts, x="Country", y="Reviews", title="Reviews per country")
    fig.update_yaxes(range=[5700, 6150])
    col2.plotly_chart(fig)

    col1, col2 = st.columns(2)

    brand_price = df.groupby("Brand")["Price_usd"].mean().reset_index().sort_values("Price_usd")
    col1.plotly_chart(px.bar(brand_price, x="Brand", y="Price_usd", title="Average price by brand (USD)"))

    rating_counts = df["Rating"].value_counts().reset_index()
    rating_counts.columns = ["Rating", "Reviews"]
    col2.plotly_chart(px.bar(rating_counts, x="Rating", y="Reviews", title="How customers rated phones"))

    st.subheader("Key findings")
    st.write("- Price is the main difference between brands: Realme is the cheapest "
             "and Apple is the most expensive.")
    st.write("- Ratings are almost the same for every brand and country (about 3.1 out of 5).")
    st.write("- A higher price does not mean a higher rating.")


def show_segments():
    st.header("Phone Segments (K-Means)")
    st.write("The 22 phone models are grouped into 4 segments by price and ratings.")

    summary = products.groupby("Segment").agg({"Model": "count", "Reviews": "sum",
                                               "Price_usd": "mean", "Rating": "mean"})
    summary = summary.reindex(SEGMENT_NAMES).round(2)
    summary.columns = ["Phones", "Reviews", "Average price (USD)", "Average rating"]
    summary["Average price (" + currency + ")"] = (summary["Average price (USD)"] * rate).round(2)
    st.dataframe(summary)

    col1, col2 = st.columns(2)

    fig = px.scatter(products, x="Price_usd", y="Rating", color="Segment", hover_name="Model",
                     color_discrete_map=SEGMENT_COLORS, title="Phones by price and rating")
    fig.update_yaxes(range=[3.0, 3.3])
    col1.plotly_chart(fig)

    sorted_products = products.sort_values("Price_usd")
    col2.plotly_chart(px.bar(sorted_products, x="Price_usd", y="Model", color="Segment",
                             color_discrete_map=SEGMENT_COLORS, orientation="h",
                             title="Average price of every model"))

    st.subheader("Look inside a segment")
    chosen = st.selectbox("Choose a segment", SEGMENT_NAMES)
    part = products[products["Segment"] == chosen][["Brand", "Model", "Price_usd", "Rating"]].round(2)
    part = add_local_price(part, country)
    st.dataframe(part.sort_values("Price_usd").reset_index(drop=True))

    st.subheader("What the segments tell us")
    st.write("- **Budget:** Realme, Xiaomi and Motorola (8 phones and about 43% of reviews).")
    st.write("- **Mid-range:** OnePlus.")
    st.write("- **Upper mid-range:** Google and Samsung.")
    st.write("- **Premium:** Apple.")
    st.write("- The segments are mainly price groups. Ratings are almost the same "
             "in every segment, so a higher price does not mean happier customers.")


# ---------------------------------------------------------
#                       Prediction tab
# ---------------------------------------------------------
def show_similar():
    st.subheader("Find Similar Phones")

    model_name = st.selectbox("Choose a phone", sorted(products["Model"].tolist()))
    top_n = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)
    same_segment = st.checkbox("Only show phones from the same segment")

    chosen = products[products["Model"] == model_name].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Segment", chosen["Segment"])
    col2.metric("Average price (USD)", round(chosen["Price_usd"], 2))
    col3.metric("Average price (" + currency + ")", round(chosen["Price_usd"] * rate, 2))
    col4.metric("Average rating", round(chosen["Rating"], 2))

    result = add_local_price(recommend_similar(model_name, top_n, same_segment), country)

    if len(result) == 0:
        st.warning("No phones found.")
    else:
        st.subheader("Recommended phones")
        st.dataframe(result)
        st.plotly_chart(px.bar(result, x="Model", y="Price_usd", color="Segment",
                               color_discrete_map=SEGMENT_COLORS,
                               title="Price of the recommended phones (USD)"))

    st.caption("Scores are close to 1 for all phones because ratings are almost the same. "
               "The order matters more than the number.")


def show_budget():
    st.subheader("Find a Phone by Budget")

    col1, col2 = st.columns(2)
    budget = col1.slider("Maximum budget (USD)", 300, 1500, 700, step=10)
    min_rating = col2.slider("Minimum rating", 1.0, 5.0, 3.0, step=0.05)
    st.write("Maximum budget = " + str(round(budget * rate, 2)) + " " + currency)

    priority_options = ["Rating", "Battery_life_rating", "Camera_rating",
                        "Performance_rating", "Design_rating", "Display_rating"]
    brand_options = ["All"] + sorted(products["Brand"].unique().tolist())

    col1, col2, col3 = st.columns(3)
    priority = col1.selectbox("What matters most?", priority_options)
    brand = col2.selectbox("Brand", brand_options)
    top_n = col3.slider("How many results?", 1, 10, 5)

    result = add_local_price(recommend_by_budget(budget, min_rating, priority, brand, top_n), country)

    if len(result) == 0:
        st.warning("No phone matches. Try a higher budget or a lower minimum rating.")
    else:
        st.subheader("Matching phones")
        st.dataframe(result)
        st.plotly_chart(px.scatter(result, x="Price_usd", y="Rating", color="Segment",
                                   hover_name="Model", color_discrete_map=SEGMENT_COLORS,
                                   title="Matching phones by price and rating"))


def show_new_phone():
    st.subheader("Predict the Segment of a New Phone")

    col1, col2 = st.columns(2)
    price = col1.number_input("Price (USD)", min_value=100, max_value=2000, value=700, step=10)
    rating = col2.slider("Overall rating", 1.0, 5.0, 3.1, step=0.01)
    st.write("Price = " + str(round(price * rate, 2)) + " " + currency)

    col1, col2, col3, col4, col5 = st.columns(5)
    battery = col1.slider("Battery life rating", 1.0, 5.0, 2.7, step=0.01)
    camera = col2.slider("Camera rating", 1.0, 5.0, 2.7, step=0.01)
    performance = col3.slider("Performance rating", 1.0, 5.0, 2.7, step=0.01)
    design = col4.slider("Design rating", 1.0, 5.0, 2.7, step=0.01)
    display = col5.slider("Display rating", 1.0, 5.0, 2.7, step=0.01)

    top_n = st.slider("How many similar phones?", 1, 10, 5)

    if st.button("Predict"):
        segment = predict_segment(price, rating, battery, camera, performance, design, display)
        st.success("This phone belongs to the " + segment + " segment.")

        part = products[products["Segment"] == segment]
        st.write("Phones in the " , segment , "segment cost between" , str(round(part["Price_usd"].min())),"USD and", str(round(part["Price_usd"].max())), "USD", 
                 "(",str(round(part["Price_usd"].min() * rate)), "to", str(round(part["Price_usd"].max() * rate)), " ", currency,")")
        st.write("Brands : ", ", ".join(sorted(part["Brand"].unique())))

        result = recommend_for_new_phone(price, rating, battery, camera, performance, design, display, top_n)
        result = add_local_price(result, country)
        st.subheader("Most similar phones")
        st.dataframe(result)
        st.plotly_chart(px.bar(result, x="Model", y="Price_usd", color="Segment",
                               color_discrete_map=SEGMENT_COLORS,
                               title="Price of the similar phones (USD)"))


def show_prediction():
    st.header("Prediction")

    option = st.radio("What do you want to do?",
                      ["Find Similar Phones", "Find a Phone by Budget",
                       "Predict the Segment of a New Phone"], horizontal=True)
    st.divider()

    if option == "Find Similar Phones":
        show_similar()
    elif option == "Find a Phone by Budget":
        show_budget()
    else:
        show_new_phone()


# ---------------------------------------------------------
#                       Main
# ---------------------------------------------------------
st.title("Mobile Product Segmentation and Recommendation System")

tab1, tab2 = st.tabs(["Dashboard", "Prediction"])

with tab1:
    show_overview()
    st.divider()
    show_segments()

with tab2:
    show_prediction()
