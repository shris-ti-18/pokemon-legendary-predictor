# 🎮 Pokémon Legendary Predictor

An end-to-end Machine Learning web application that predicts whether a Pokémon is **Legendary** or **Normal** based entirely on its base battle stats. This project covers the entire data science pipeline—from exploratory data analysis (EDA) and data wrangling to model training, evaluation, and cloud deployment.

🔗 **Live Demo:** [Launch the App on Streamlit Community Cloud](https://pokemon-legendary-predictor-mxdny4ko6kmarwt2laltkp.streamlit.app/)

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning:** Scikit-learn
* **Web Framework & Deployment:** Streamlit

---

## 📊 Exploratory Data Analysis (EDA)

Before building the predictive model, the dataset was analyzed to explore feature distributions and statistical anomalies:

### 1. Feature Correlation

The correlation heatmap highlights how various base metrics weigh against each other. Stronger total base values show explicit links to performance spikes in individual sub-stats.

*As visualized in `Screenshot 2026-06-21 143711.png`:*

* Features like `Sp. Atk`, `Attack`, and `Speed` demonstrate noticeable correlations with legendary classification status.

### 2. Class Variance

*As visualized in `Screenshot 2026-06-21 143207.png`:*

* Plotting `Attack` vs. `Speed` clearly exhibits how Legendary entries populate higher-tier stat clusters compared to standard, everyday Pokémon.

---

## 🤖 Model Evaluation & Metrics

The project uses a **Random Forest Classifier** trained on an $80/20$ data split. Given that the dataset is highly imbalanced (with far fewer Legendary Pokémon than regular ones), the model was carefully evaluated using a classification report to track performance across individual classes.

### Performance Breakdown

*As reported in `Screenshot 2026-06-21 143753_2.png`:*

| Target Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **False (Normal)** | 0.97 | 0.99 | 0.98 | 150 |
| **True (Legendary)** | 0.75 | 0.60 | 0.67 | 10 |
| **Overall Accuracy** |  |  | **0.96** | 160 |
| **Macro Average** | 0.86 | 0.79 | 0.82 | 160 |
| **Weighted Average** | 0.96 | 0.96 | 0.96 | 160 |

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/shris-ti-18/pokemon-legendary-predictor.git
cd pokemon-legendary-predictor

```

### 2. Install Dependencies

Make sure you have your virtual environment active, then install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit

```

### 3. Run the Streamlit Application

```bash
streamlit run app.py

```

---

## 💡 How It Works

1. **Input Parameters:** The user inputs a Pokémon's base values (`HP`, `Attack`, `Defense`, `Sp. Atk`, `Sp. Def`, and `Speed`) into the web interface.
2. **Model Inference:** On clicking **Predict**, the inputs are formatted into a matching matrix row and evaluated by the pre-trained Random Forest model.
3. **Result Output:** The system instantly returns whether the custom configuration meets the classification boundary for a **⭐ Legendary Pokémon** or stays within standard **🐾 Non-Legendary** limits, alongside confidence probability metrics.



