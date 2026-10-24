import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cosmetics Price Predictor", page_icon="💄")

model = joblib.load("models/model.joblib")
meta = json.load(open("models/metadata.json"))
cols = meta["columns"]

st.title("💄 Cosmetics Price Predictor")
st.caption(f"{meta['model_name']} | MAE ${meta['metrics']['MAE']:.1f} | R2 {meta['metrics']['R2']:.2f}")

label = st.selectbox("Product type", meta["labels"])
brand = st.selectbox("Brand", meta["brands"])
rank = st.slider("Rating", 0.0, 5.0, 4.2, 0.1)
n_ing = st.slider("Number of ingredients", 1, 80, 25)

st.write("Skin type")
skin_names = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]
c = st.columns(5)
skin = {s: int(c[i].checkbox(s)) for i, s in enumerate(skin_names)}

chosen = st.multiselect("Key ingredients", meta["ingredients"])

row = {col: 0 for col in cols}
row["Rank"] = rank
row["n_ingredients"] = n_ing
row.update(skin)
row[f"Label_{label}"] = 1
row[f"Brand_grp_{brand}"] = 1
for ing_name in chosen:
    row[f"ing_{ing_name}"] = 1

x = pd.DataFrame([row], columns=cols)
price = float(np.expm1(model.predict(x)[0]))

st.subheader("Predicted price")
st.success(f"${price:.2f}")
st.caption(f"Typical error is about ${meta['metrics']['MAE']:.0f}.")
