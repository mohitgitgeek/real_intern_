# real_intern_
My First self-project.

SCREENSHOT :
![image](https://github.com/user-attachments/assets/6b1aed4a-60a7-4e39-ba60-677aafed7e92)

A full Streamlit app: **frontend + backend** (Streamlit), an **ML model**, and **DB storage**.

- **Frontend + backend:** Streamlit UI with two tabs — *Analyze* and *History*.
- **Machine Learning:** TF-IDF vectoriser + scikit-learn **RandomForestClassifier** classify an internship
  email as legitimate or fraudulent, with a confidence score and keyword-based risk factors. Tokenisation is
  dependency-free (no NLTK `punkt` download required).
- **Database:** every analysis is saved to a local **SQLite** database (`intern_history.db`, see `db.py`);
  the History tab shows past results, summary stats, and a clear-history button.

STEPS :
1) Clone this repo.
2) Install dependencies and run:
```sh
pip install -r requirements.txt
streamlit run intern_detect.py
```

