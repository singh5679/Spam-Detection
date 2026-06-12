# 📩 SMS Spam Detection

A machine learning-powered **SMS Spam Classifier** web application built with **Streamlit**. It predicts whether a given SMS message is **Spam** or **Ham (Not Spam)** in real time using NLP techniques.

---

## 🌐 Live Demo

> Run locally using the steps below — see [Installation](#installation)

---

## 📸 App Preview
![alt text](<Screenshot 2025-07-15 122412.png>)
![alt text](<Screenshot 2025-07-15 122559.png>)
![alt text](<Screenshot 2025-07-15 134633.png>)

> *(Add a screenshot of your Streamlit app here)*

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Author](#author)

---

## 🧠 Overview

This project is an end-to-end **SMS Spam Detection System** that:
1. Preprocesses raw SMS text (cleaning, tokenization, stopword removal, stemming)
2. Converts text to numerical features using **TF-IDF Vectorization**
3. Classifies messages as **Spam** or **Ham** using a trained ML model
4. Serves predictions via an interactive **Streamlit** web interface

---

## ✨ Features

- 🔤 Real-time SMS spam prediction
- 🧹 NLP preprocessing pipeline (NLTK)
- 📊 TF-IDF based feature extraction
- 🌐 Clean, interactive Streamlit UI
- ⚡ Instant results on user input

---

## 🛠️ Tech Stack

| Category       | Tools                        |
|----------------|------------------------------|
| Language       | Python 3.x                   |
| Web Framework  | Streamlit                    |
| ML Library     | Scikit-learn                 |
| NLP            | NLTK                         |
| Data Handling  | Pandas, NumPy                |
| Serialization  | Pickle                       |

---

## 📊 Dataset

- **Source:** [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size:** 5,574 SMS messages
- **Classes:** `spam` / `ham`
- **Split:** 80% Training / 20% Testing

---

## ⚙️ How It Works

```
User Input (SMS Text)
        ↓
Text Preprocessing (Lowercase → Tokenize → Remove Stopwords → Stem)
        ↓
TF-IDF Vectorization
        ↓
ML Model Prediction
        ↓
Output: SPAM 🚨 or HAM ✅
```

---

## 💻 Installation

```bash
# 1. Clone the repository
git clone https://github.com/singh5679/Spam-Detection.git
cd Spam-Detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`, type any SMS message, and click **Predict** to see if it's spam or ham.

---

## 📈 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~98%   |
| Precision | ~99%   |
| Recall    | ~90%   |
| F1-Score  | ~94%   |

> Update these numbers with your actual model evaluation results.

---

## 📁 Project Structure

```
Spam-Detection/
│
├── app.py                  # Streamlit web application
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # Fitted TF-IDF Vectorizer
├── spam.csv                # Dataset
├── spam_detection.ipynb    # Model training notebook
├── requirements.txt
└── README.md
```

> ⚠️ Adjust the file names above to match your actual project structure.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and submit a pull request.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Himanshu Singh**  
B.Tech 2026 | AI/ML & Full-Stack Developer  
📍 Lucknow, India

[![GitHub]](https://github.com/singh5679)
[![LinkedIn]](https://www.linkedin.com/in/himanshu-singh-3b6662283/)

---

> ⭐ If you found this project helpful, give it a star — it means a lot!
