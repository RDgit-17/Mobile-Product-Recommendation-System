# Mobile Product Segmentation and Recommendation System

An end-to-end data analytics and machine learning project that cleans global smartphone review data, segments phone models into market tiers with **K-Means clustering**, and recommends similar phones using **cosine similarity**.

> Domain: E-Commerce Analytics & Intelligent Product Recommendation Systems

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key EDA Findings](#key-eda-findings)
- [Segmentation Results](#segmentation-results)
- [Recommendation System](#recommendation-system)
- [Tech Stack](#tech-stack)


---

## Project Overview

The smartphone market produces a large amount of structured data (prices, ratings, reviews) that is hard to use directly. This project turns raw review data into:

1. A **clean dataset** (missing values, duplicates and data types handled).
2. **EDA insights** on brands, countries, prices and ratings.
3. **Four market segments** (Budget, Mid-range, Upper mid-range, Premium) found with K-Means.
4. A **similarity-based recommender** that suggests phones by selected model, by a new phone's specs, or by budget.
5. An **interactive Streamlit app** for exploring segments and getting recommendations.

## Dataset

**Global Mobile Reviews Dataset**: 50,000 reviews across 22 columns.

| Item | Raw | After cleaning |
|---|---|---|
| Reviews | 50,000 | 47,432 |
| Columns | 22 | 18 |
| Brands | 7 | 7 |
| Phone models | 22 | 22 |
| Countries | 8 | 8 |

*Brands :* Apple, Samsung, Google, OnePlus, Xiaomi, Realme, Motorola. 

*Countries :* USA, UK, Canada, Australia, Germany, India, UAE, Brazil.

*Main columns used :* `Price_usd`, `Rating`, `Battery_life_rating`, `Camera_rating`, `Performance_rating`, `Design_rating`, `Display_rating`, plus `Brand`, `Model`, `Country`, `Currency` and `Exchange_rate_to_usd`.

## Project Workflow

### 1. Data Cleaning 
- Checked for duplicates (none found) and null values (about 4.9% in price, rating and sentiment).
- Dropped columns that add no value: `review_id`, `customer_name`, `language`.
- Cleaned `price_local` (removed currency symbols and commas), dropped rows with both prices missing, and filled single missing prices using the exchange rate.
- Dropped rows with a missing `rating` rather than imputing it.
- Dropped `sentiment` because it was inconsistent with the star rating.
- Verified that every model maps to one brand and every currency to one country.
- Fixed data types (dates, integers) and standardised column names.

### 2. Exploratory Data Analysis 
Outlier (IQR) and skewness checks, brand and country distributions, top and low-rated models, price vs rating, age groups, verified purchases, review sources and time trends.

### 3. Feature Engineering and Scaling 
- Reviews are aggregated to **one row per phone model** (mean price and ratings, plus review count).
- `Price_usd` is min-max scaled; the ratings are scaled from the 1 to 5 range to 0 to 1.

### 4. Clustering
- K-Means with **k = 4** (`n_init=10`, `random_state=42`).
- Clusters are named from cheapest to most expensive by average price.

### 5. Recommendation System
Cosine similarity on the scaled feature vectors (see below).

### 6. Streamlit App
Interactive app for segment insights and recommendations.

## Key EDA Findings

1. The data is balanced: each brand has about 6,700 to 6,850 reviews and each country about 5,800 to 6,100.
2. **Price is what separates brands.** Realme is cheapest (about $393 average) and Apple is the most expensive (about $1,104).
3. **Ratings are almost identical** across brands, countries and models (about 3.1 out of 5). The best and worst models differ by only about 0.12 points.
4. **A higher price does not mean a higher rating.** The price-rating correlation is close to zero.
5. The overall rating moves together with the five sub-ratings, and higher-rated reviews get more helpful votes.
6. About 73% of reviewers are aged 35 or younger.
7. No real outliers or data errors were found, so no rows were removed for being outliers.

## Segmentation Results

| Segment | Models | Avg price (USD) | Models in segment |
|---|---|---|---|
| Budget | 8 | about $457 | Edge 50, Moto G Power, Razr 40, Realme 12 Pro, Realme Narzo 70, Mi 13 Pro, Poco X6, Redmi Note 13 |
| Mid-range | 3 | about $673 | OnePlus 11R, OnePlus 12, OnePlus Nord 3 |
| Upper mid-range | 7 | about $859 | Pixel 6, Pixel 7a, Pixel 8, Galaxy A55, Galaxy Note 20, Galaxy S24, Galaxy Z Flip |
| Premium | 4 | about $1,104 | iPhone 13, iPhone 14, iPhone 15 Pro, iPhone SE |


**Interpretation :** because ratings are nearly identical everywhere, the segments are driven almost entirely by **price**, which matches the EDA finding above. In this dataset, a phone's tier tells you what it costs, not how well it is rated.

## Recommendation System


| Function | What it does |
|---|---|
| `recommend_similar(model_name, top_n, same_segment_only)` | Phones most similar to a chosen model |
| `recommend_for_new_phone(price, rating, battery, camera, performance, design, display, top_n)` | Phones closest to a set of specs entered by the user |
| `recommend_by_budget(budget, min_rating, priority, brand, top_n)` | Filter by budget, minimum rating and brand, ranked by a chosen rating |
| `predict_segment(...)` | Predicts the segment of a new phone from its specs |
| `add_local_price(result, country)` | Converts USD prices to a country's local currency |


## Tech Stack

Python, Pandas, NumPy, Scikit-learn (K-Means, cosine similarity), Matplotlib, Seaborn, Plotly, Streamlit, Jupyter.




