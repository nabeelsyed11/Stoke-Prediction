# 🩺 Stroke Prediction App

An interactive web application built with **Streamlit** to predict the risk of stroke based on patient health details.  
The model uses a **Random Forest Classifier** trained on the [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset).

---

## 📌 Features
- User-friendly interface for entering patient details.
- Automatic data preprocessing (missing value imputation, label encoding).
- Stroke risk prediction with probability score.
- Fun GIF animation for user engagement.
- Model training done on application load.

---

## 🛠 Tech Stack
- **Python**
- **Streamlit** for web interface
- **Pandas** & **NumPy** for data handling
- **scikit-learn** for model training
- **Random Forest Classifier** as ML model

---

## 🚀 Installation & Usage
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add dataset**
   - Place `train.csv` in the project root directory.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The app will open automatically in your default browser, or visit:
     ```
     http://localhost:8501
     ```

---

## 📂 Project Structure
```
.
├── app.py               # Main Streamlit app
├── train.csv            # Dataset (not included in repo)
├── requirements.txt     # Required dependencies
└── README.md            # Project documentation
```

---

## 📊 Model Details
- **Algorithm:** Random Forest Classifier
- **Data preprocessing:**
  - Missing BMI values filled with mean.
  - Label encoding for categorical features.
- **Split:** 80% training, 20% testing.
- **Target Variable:** `stroke`

---

## Project Link
[https://stoke-prediction-japxndvuiyuqppd4epietu.streamlit.app/]

---
## ⚠ Disclaimer
This tool is for **educational purposes only** and should not be used for real medical diagnosis.

---

## 👨‍💻 Author
- Developed by [Syed Nabeel Ahmed]
- 📧 Contact: [nabeelahmedna7860@gmail.com]
