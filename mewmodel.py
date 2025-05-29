import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import joblib

# Загрузка данных
df = pd.read_csv('df_russian_articles.csv')

# Удаление ненужных столбцов и дубликатов
df.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "lang"], inplace=True)
df = df.drop_duplicates()
df = df.dropna()

# Функция для очистки текста
def clean_text(text):
    import re
    from simplemma import lemmatize
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('russian') + stopwords.words('english'))
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", "", text)
    text = text.lower()
    words = text.split()
    cleaned_words = [lemmatize(word, 'ru') for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Очистка заголовков
df['clean_title'] = df['title'].apply(clean_text)

# Кодирование категорий
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['category'] = encoder.fit_transform(df['category'])

# Сохранение LabelEncoder
joblib.dump(encoder, 'label_encoder.pkl')

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(df['clean_title'], df['category'], test_size=0.2, random_state=42)

# Векторизация заголовков
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Сохранение векторайзера
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Сохранение TF-IDF матрицы всех заголовков для рекомендаций
all_titles_tfidf = vectorizer.transform(df['clean_title'])
joblib.dump(all_titles_tfidf, 'all_titles_tfidf.pkl')
joblib.dump(df[['title', 'category']], 'articles_data.pkl')  # Сохраняем заголовки и категории

# Обучение моделей
models = {
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=13, weights='distance'),
    "MultinomialNB": MultinomialNB(),
    "CatBoostClassifier": CatBoostClassifier(iterations=100, verbose=False),
    "LogisticRegression": LogisticRegression(penalty='l2', C=10),
    "SVC": SVC(kernel='linear')
}

# Обучение и сохранение каждой модели
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    joblib.dump(model, f'{name}.pkl')
    print(f"Модель {name} обучена и сохранена в {name}.pkl")

print("Обучение завершено, все модели и данные сохранены.")