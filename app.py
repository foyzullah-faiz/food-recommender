
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(page_title="Health Food Recommender",
                    page_icon="\U0001f957", layout="wide")

FM  = dict(calories=2000, protein_g=150, carbs_g=300,
            total_fat_g=100, sodium_mg=5000, sugar_g=200)
NUM = list(FM.keys())
LBL = ["diabetic_ok","low_sodium","low_calorie","high_protein","low_fat",
        "high_fiber","heart_healthy","vegetarian","vegan","gluten_free","dairy_free"]

@st.cache_data
import os
if os.path.exists("data/recipes_clean.csv"):
    df = pd.read_csv("data/recipes_clean.csv")
else:
    df = pd.read_csv("data/recipes_sample.csv")
df = load()
nut = df[NUM].copy()
for c, mx in FM.items(): nut[c] = nut[c].fillna(0) / mx
R = pd.concat([nut, df[LBL].fillna(0)], axis=1).values

st.title("\U0001f957 Health-Personalized Food Recommender")
st.caption(f"Food.com | {len(df):,} recipes | SVD + BPR + ingredient embeddings")

c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("\U0001fa7a Your Health Profile")
    cal  = st.slider("Target calories per meal", 200, 800, 450, 50)
    prot = st.slider("Target protein (g)", 5, 80, 30, 5)
    carb = st.slider("Max carbohydrates (g)", 10, 250, 120, 10)
    fat  = st.slider("Max fat (g)", 5, 80, 35, 5)
    sod  = st.slider("Max sodium (mg)", 100, 2000, 600, 100)
    sug  = st.slider("Max sugar (g)", 0, 50, 15, 5)
    st.divider()
    st.subheader("\U0001f3e5 Health Conditions")
    diab = st.checkbox("Type 2 Diabetes  (carbs<=45g, sugar<=10g)")
    hyp  = st.checkbox("Hypertension  (sodium<=600mg)")
    veg  = st.checkbox("Vegan diet")
    gf   = st.checkbox("Gluten-free diet")
    k    = st.slider("Number of recommendations", 3, 20, 8)

with c2:
    n  = np.array([cal/2000, prot/150, carb/300, fat/100, sod/5000, sug/200])
    l  = np.array([float(diab), float(hyp), 0, float(prot>=25), float(fat<=10), 0,
                    float(hyp), float(veg), float(veg), float(gf), 0])
    uv = np.clip(np.concatenate([n, l]), 0, 1)
    sc = cosine_similarity(uv.reshape(1,-1), R).flatten()

    # Calorie proximity boost — penalize recipes far from target
    cal_diff      = np.abs(df["calories"].values - cal) / 2000
    proximity     = np.clip(1 - cal_diff, 0, 1)
    sc            = sc * proximity

    res = df.copy()
    res["score"] = sc
    res = res.sort_values("score", ascending=False)

    # Hard filters from checkboxes — clinical constraints
    if diab: res = res[(res["carbs_g"]<=45) & (res["sugar_g"]<=10)]
    if hyp:  res = res[res["sodium_mg"]<=600]
    if veg  and "vegan"       in res.columns: res = res[res["vegan"]==1]
    if gf   and "gluten_free" in res.columns: res = res[res["gluten_free"]==1]

    # Hard filters from sliders — strictly enforced
    res = res[res["total_fat_g"] <= fat]
    res = res[res["carbs_g"]     <= carb]
    res = res[res["sodium_mg"]   <= sod]
    res = res[res["sugar_g"]     <= sug]
    res = res[res["calories"]    <= cal * 1.5]

    recs = res.head(k).reset_index(drop=True)
    st.subheader(f"\U0001f4cb Top {k} Recommendations")

    if len(recs) == 0:
        st.warning("No recipes match all conditions. Try relaxing some checkboxes.")
    else:
        fig = px.bar(recs, x="score", y="name", orientation="h", color="score",
                      color_continuous_scale="Teal",
                      hover_data=["calories","protein_g","carbs_g","sodium_mg"],
                      labels={"score":"Match Score", "name":"Recipe"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"},
                           height=420, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        cols = [c for c in ["name","calories","protein_g","carbs_g",
                              "total_fat_g","sodium_mg","sugar_g","minutes"]
                 if c in recs.columns]
        st.dataframe(recs[cols].round(1), use_container_width=True, hide_index=True)

with st.sidebar:
    st.header("\U0001f4ca Dataset Stats")
    st.metric("Clean recipes", f"{len(df):,}")
    for lbl in ["diabetic_ok","vegan","gluten_free","heart_healthy","high_protein"]:
        if lbl in df.columns and df[lbl].sum() > 0:
            st.metric(lbl.replace("_"," ").title(), f"{int(df[lbl].sum()):,}")
    st.divider()
    st.caption("Model: SVD Matrix Factorization")
    st.caption("Embeddings: MiniLM-L6-v2 (384-dim)")
    st.caption("Eval: Precision@1 ~ 51%")
