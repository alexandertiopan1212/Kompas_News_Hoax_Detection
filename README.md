# 🕵️‍♂️ Kompas News Hoax Detection

> A machine learning and deep learning-based system to automatically classify factual vs hoax news articles using cleaned data from Kompas and TurnBackHoax.

## 📌 Project Overview

This project aims to detect hoaxes in news articles by leveraging Natural Language Processing (NLP) techniques and classification models. We use a hybrid dataset derived from **Kompas.com** (factual news) and **TurnBackHoax.id** (hoaxes) to build and evaluate models including **Random Forest** and **Convolutional Neural Network (CNN)**.

---

## 📂 Dataset

| Dataset File                       | Description                                 |
|-----------------------------------|---------------------------------------------|
| `dataset_kompas_4k_cleaned.xlsx`  | Cleaned Kompas news articles (label: 0)     |
| `dataset_turnbackhoax_10_cleaned.xlsx` | Cleaned TurnBackHoax news (label: 1)    |

These two datasets are concatenated into a single DataFrame `news_df` with labels: `0` (factual) and `1` (hoax).

---

## ⚙️ Preprocessing Steps

1. **Lower Casing**
2. **Character Cleaning**
   - Remove punctuation, numbers, ASCII/Unicode noise, and extra spaces.
3. **Stopword Removal**
   - Uses `stopwordbahasa.csv` as Indonesian stopwords list.
4. **Stemming**
   - Using [Sastrawi](https://github.com/har07/PySastrawi) stemmer for Bahasa Indonesia.
5. **Tokenizing**
   - Tokenize with `nltk.tokenize.word_tokenize`.
6. **Vectorization**
   - CountVectorizer (Unigram) to convert text into feature vectors.

---

## 🤖 Models & Performance

### 1. 🌲 Random Forest

- **Split:** 30% train / 70% test
- **Accuracy:** `99.67%`
- **Precision:** `99.64%`
- **Recall:** `99.89%`
- **F1-Score:** `99.76%`

✅ **Strengths:**
- Fast training
- High accuracy
- Interpretable (feature importances)

---

### 2. 🧠 Convolutional Neural Network (CNN)

- 1D Convolution + Global Max Pooling + Dense Layer
- **Test Accuracy:** `68.1%`
- **Precision:** `46.3%`
- **Recall:** `68.1%`
- **Loss:** `0.60`

⚠️ **Notes:**
- CNN performed moderately.
- Requires more data or hyperparameter tuning for improvements.

---

## 🔍 Comparison: Random Forest vs CNN

| Metric       | Random Forest | CNN       |
|--------------|----------------|-----------|
| Accuracy     | ✅ High (99.7%) | ⚠️ Moderate (68.1%) |
| Precision    | ✅ High         | ⚠️ Low (46.3%) |
| Recall       | ✅ High         | ⚠️ Moderate |
| Interpretability | ✅ Easy     | ❌ Black-box |
| Speed        | ✅ Fast         | ❌ Slower to train |

**Conclusion:**  
Random Forest significantly outperforms CNN in this case due to the nature of the dataset and the limited sample size. However, CNN still offers promise with more training data and parameter optimization.

---

## 🛠️ Libraries Used

- **Pandas, NumPy**
- **Sastrawi (Indonesian Stemmer)**
- **scikit-learn**
- **NLTK**
- **TensorFlow / Keras**

---

## 📊 Evaluation Metrics

- `Accuracy`
- `Precision`
- `Recall`
- `F1 Score`
- `Loss (CNN only)`

---

## 👤 Author

**Alexander Tiopan**  
GitHub: [alexandertiopan1212](https://github.com/alexandertiopan1212)

---

## 📁 Repository Structure

Kompas_News_Hoax_Detection/
│
├── dataset_kompas_4k_cleaned.xlsx
├── dataset_turnbackhoax_10_cleaned.xlsx
├── stopwordbahasa.csv
├── main.ipynb
└── README.md


---

## 📬 License

This project is under the **MIT License**.

---
