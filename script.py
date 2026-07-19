# =========================================
# Kirana Shop ML Recommendation API
# =========================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

# Apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Similar Products ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


# =====================================================
# ------------------ 1️⃣ APRIORI ----------------------
# =====================================================

class TransactionRequest(BaseModel):
    transactions: List[List[str]]
    product: str


def generate_rules(transactions):

    if not transactions:
        return None

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)

    if df.empty:
        return None

    # Lower support for small dataset
    frequent_items = apriori(df, min_support=0.02, use_colnames=True)

    if frequent_items.empty:
        return None

    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=0.1
    )

    if rules.empty:
        return None

    print("Generated Rules:")
    print(rules[['antecedents', 'consequents', 'confidence']])

    return rules


def recommend_from_rules(rules, product_name):

    if rules is None:
        return []

    recommendations = []

    for _, row in rules.iterrows():

        antecedents = [x.lower().strip() for x in row["antecedents"]]
        consequents = [x.lower().strip() for x in row["consequents"]]

        # product -> consequent
        if product_name in antecedents:
            recommendations.extend(consequents)

        # consequent -> product (reverse rule)
        if product_name in consequents:
            recommendations.extend(antecedents)

    return list(set(recommendations))


@app.post("/recommend")
def get_recommendation(data: TransactionRequest):

    # Normalize transactions
    transactions = [
        [item.lower().strip() for item in t]
        for t in data.transactions
    ]

    product = data.product.lower().strip()

    print("Transactions received:", transactions)
    print("Product clicked:", product)

    rules = generate_rules(transactions)

    result = recommend_from_rules(rules, product)

    return {
        "product_clicked": product,
        "recommended_items": result
    }


# =====================================================
# --------------- 2️⃣ SIMILAR PRODUCTS ----------------
# =====================================================

class SimilarProductRequest(BaseModel):
    product_id: int
    products: List[dict]


def get_similar_products(product_id, products):

    df = pd.DataFrame(products)

    if df.empty:
        return []

    vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_matrix = vectorizer.fit_transform(df['title'])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    if product_id not in df['id'].values:
        return []

    idx = df.index[df['id'] == product_id][0]

    similarity_scores = list(enumerate(similarity_matrix[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_similar = similarity_scores[1:6]

    similar_products = [
        df.iloc[i[0]].to_dict()
        for i in top_similar
    ]

    return similar_products


@app.post("/similar-products")
def similar_products(data: SimilarProductRequest):

    result = get_similar_products(
        data.product_id,
        data.products
    )

    return {
        "similar_products": result
    }


# =====================================================
# --------------- 3️⃣ SMART BUNDLES -------------------
# =====================================================

class BundleRequest(BaseModel):
    transactions: List[List[str]]


def generate_bundles(transactions):

    if not transactions:
        return []

    transactions = [
        [item.lower().strip() for item in t]
        for t in transactions
    ]

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)

    df = pd.DataFrame(te_array, columns=te.columns_)

    if df.empty:
        return []

    frequent_items = apriori(df, min_support=0.02, use_colnames=True)

    if frequent_items.empty:
        return []

    bundles = frequent_items[
        frequent_items['itemsets'].apply(lambda x: len(x) >= 2)
    ]

    bundle_list = []

    for _, row in bundles.iterrows():
        bundle_list.append({
            "products": list(row['itemsets']),
            "support": float(row['support'])
        })

    bundle_list = sorted(
        bundle_list,
        key=lambda x: x['support'],
        reverse=True
    )

    return bundle_list[:5]


@app.post("/bundles")
def smart_bundles(data: BundleRequest):

    result = generate_bundles(data.transactions)

    return {
        "bundles": result
    }


@app.post("/cart-recommend")
def cart_recommend(data: dict):

    cart_items = data["cart_items"]
    transactions = data["transactions"]

    rules = generate_rules(transactions)

    recommendations = set()

    for _, row in rules.iterrows():

        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])

        if set(antecedents).issubset(set(cart_items)):

            for item in consequents:
                if item not in cart_items:
                    recommendations.add(item)

    return {
        "cart_items": cart_items,
        "recommended": list(recommendations)
    }