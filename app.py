import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Pokemon Legendary Predictor",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------
# LOAD DATA
# ---------------------------

df = pd.read_csv("Pokemon.csv")

# ---------------------------
# TRAIN MODEL
# ---------------------------

X = df[
    [
        "HP",
        "Attack",
        "Defense",
        "Sp. Atk",
        "Sp. Def",
        "Speed"
    ]
]

y = df["Legendary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ---------------------------
# SIDEBAR
# ---------------------------

st.sidebar.title("⚡ Pokemon Predictor")
st.sidebar.write("Legendary Classification Model")

# ---------------------------
# TITLE
# ---------------------------

st.title("⚡ Pokemon Legendary Predictor")

st.markdown(
    """
    Predict whether a Pokemon is **Legendary**
    based on its battle statistics.
    """
)

# ---------------------------
# DASHBOARD METRICS
# ---------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Pokemon",
        len(df)
    )

with col2:
    st.metric(
        "Legendary Pokemon",
        int(df["Legendary"].sum())
    )

with col3:
    st.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

st.markdown("---")

# ---------------------------
# USER INPUTS
# ---------------------------

st.subheader("Enter Pokemon Stats")

col1, col2 = st.columns(2)

with col1:
    hp = st.number_input(
        "HP",
        min_value=1,
        max_value=255,
        value=50
    )

    attack = st.number_input(
        "Attack",
        min_value=1,
        max_value=255,
        value=50
    )

    defense = st.number_input(
        "Defense",
        min_value=1,
        max_value=255,
        value=50
    )

with col2:
    spatk = st.number_input(
        "Sp. Atk",
        min_value=1,
        max_value=255,
        value=50
    )

    spdef = st.number_input(
        "Sp. Def",
        min_value=1,
        max_value=255,
        value=50
    )

    speed = st.number_input(
        "Speed",
        min_value=1,
        max_value=255,
        value=50
    )

# ---------------------------
# PREDICT BUTTON
# ---------------------------

if st.button("Predict"):

    pokemon = [[
        hp,
        attack,
        defense,
        spatk,
        spdef,
        speed
    ]]

    prediction = model.predict(pokemon)[0]

    probabilities = model.predict_proba(pokemon)

    confidence = max(probabilities[0]) * 100

    st.markdown("---")

    if prediction:

        st.success(
            f"⭐ Legendary Pokemon\n\nConfidence: {confidence:.2f}%"
        )

        st.balloons()

    else:

        st.warning(
            f"🐾 Non-Legendary Pokemon\n\nConfidence: {confidence:.2f}%"
        )

    # -----------------------
    # USER STATS CHART
    # -----------------------

    st.subheader("Your Pokemon Stats")

    stats = pd.DataFrame(
        {
            "Stat": [
                "HP",
                "Attack",
                "Defense",
                "SpAtk",
                "SpDef",
                "Speed"
            ],
            "Value": [
                hp,
                attack,
                defense,
                spatk,
                spdef,
                speed
            ]
        }
    )

    st.bar_chart(
        stats.set_index("Stat")
    )

# ---------------------------
# FEATURE IMPORTANCE
# ---------------------------

st.markdown("---")

st.subheader("Feature Importance")

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)

# ---------------------------
# TYPE DISTRIBUTION
# ---------------------------

st.markdown("---")

st.subheader("Pokemon Type Distribution")

type_counts = df["Type 1"].value_counts()

st.bar_chart(type_counts)

# ---------------------------
# TOP 10 STRONGEST POKEMON
# ---------------------------

st.markdown("---")

st.subheader("Top 10 Strongest Pokemon")

top10 = (
    df.sort_values(
        "Total",
        ascending=False
    )[["Name", "Type 1", "Total"]]
    .head(10)
)

st.dataframe(
    top10,
    use_container_width=True
)

# ---------------------------
# DATASET PREVIEW
# ---------------------------

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# ---------------------------
# FOOTER
# ---------------------------

st.markdown("---")

st.markdown(
    """
    Built using **Python, Pandas, Scikit-Learn and Streamlit**
    """
)

