# import streamlit as st
# import joblib
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Функция для очистки текста (должна быть идентична той, что в обучении)
# def clean_text(text):
#     import re
#     from simplemma import lemmatize
#     from nltk.corpus import stopwords
#     stop_words = set(stopwords.words('russian') + stopwords.words('english'))
#     text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", "", text)
#     text = text.lower()
#     words = text.split()
#     cleaned_words = [lemmatize(word, 'ru') for word in words if word not in stop_words]
#     return " ".join(cleaned_words)

# # Загрузка сохраненных объектов
# vectorizer = joblib.load('tfidf_vectorizer.pkl')
# label_encoder = joblib.load('label_encoder.pkl')
# all_titles_tfidf = joblib.load('all_titles_tfidf.pkl')
# articles_data = joblib.load('articles_data.pkl')

# # Загрузка моделей
# models = {
#     "DecisionTreeClassifier": joblib.load('DecisionTreeClassifier.pkl'),
#     "KNeighborsClassifier": joblib.load('KNeighborsClassifier.pkl'),
#     "MultinomialNB": joblib.load('MultinomialNB.pkl'),
#     "CatBoostClassifier": joblib.load('CatBoostClassifier.pkl'),
#     "LogisticRegression": joblib.load('LogisticRegression.pkl'),
#     "SVC": joblib.load('SVC.pkl')
# }

# # Заголовок приложения
# st.title("🔍 Определить раздел науки и найти похожие статьи")

# # Ввод заголовка статьи
# user_input = st.text_input("Введите заголовок вашей статьи:")

# if st.button("Анализировать"):
#     if user_input:
#         # Очистка и векторизация введенного заголовка
#         cleaned_input = clean_text(user_input)
#         input_tfidf = vectorizer.transform([cleaned_input]).toarray()

#         # Предсказание категории
#         results = {}
#         for model_name, model in models.items():
#             predicted_class = model.predict(input_tfidf)[0]
#             results[model_name] = label_encoder.inverse_transform([predicted_class])[0]

#         # Вывод предсказанных категорий
#         st.write("### Предсказанные разделы науки:")
#         for model, category in results.items():
#             st.write(f"{model}: {category}")

#         # Поиск похожих статей
#         similarities = cosine_similarity(input_tfidf, all_titles_tfidf)[0]
#         top_indices = np.argsort(similarities)[::-1][:5]  # Топ-5 по убыванию сходства

#         st.write("### Топ-5 похожих статей:")
#         for idx in top_indices:
#             similarity_score = similarities[idx]
#             if similarity_score > 0:  # Выводим только статьи с ненулевым сходством
#                 title = articles_data.iloc[idx]['title']
#                 category = label_encoder.inverse_transform([articles_data.iloc[idx]['category']])[0]
#                 st.write(f"- **{title}** (Раздел: {category}, Сходство: {similarity_score:.4f})")
#             else:
#                 st.write("Нет достаточно похожих статей в наборе данных.")
#                 break

#     else:
#         st.warning("Пожалуйста, введите заголовок статьи.")

# st.text("*где информатика - 0, математика - 1, физика - 2, химия - 3")
# st.markdown("""
#     ---
#     Сделано с ❤️ и с использованием Streamlit и машинного обучения.
# """)


import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
from simplemma import lemmatize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# import locale
# locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8') 

# Функция для очистки текста
def clean_text(text):
    stop_words = set(stopwords.words('russian') + stopwords.words('english'))
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", "", text)
    text = text.lower()
    words = text.split()
    cleaned_words = [lemmatize(word, 'ru') for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Функция для генерации ключевых слов
def get_top_keywords(input_tfidf, vectorizer, top_n=5):
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = input_tfidf.toarray()[0]
    top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
    return [feature_names[idx] for idx in top_indices if tfidf_scores[idx] > 0]

# Функция для создания облака слов
def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Загрузка сохраненных объектов
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
all_titles_tfidf = joblib.load('all_titles_tfidf.pkl')
articles_data = joblib.load('articles_data.pkl')
articles = pd.read_csv('df_russian_articles.csv')
articles.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "lang"], inplace=True)

# Загрузка моделей
models = {
    "DecisionTreeClassifier": joblib.load('DecisionTreeClassifier.pkl'),
    "KNeighborsClassifier": joblib.load('KNeighborsClassifier.pkl'),
    "MultinomialNB": joblib.load('MultinomialNB.pkl'),
    "CatBoostClassifier": joblib.load('CatBoostClassifier.pkl'),
    "LogisticRegression": joblib.load('LogisticRegression.pkl'),
    "SVC": joblib.load('SVC.pkl')
}

# Заголовок приложения
st.set_page_config(page_title="Анализ статей", page_icon="🔍")
st.title("🔍 Определение научного раздела и поиск похожих статей")
st.subheader("Введите заголовок вашей статьи для анализа")

# Вывод датасета статей
st.write("### Датасет статей:")
st.dataframe(articles)

# Ввод заголовка статьи
user_input = st.text_input("Введите заголовок вашей статьи:")

if st.button("Анализировать"):
    if user_input:
        # Очистка и векторизация введенного заголовка
        cleaned_input = clean_text(user_input)
        input_tfidf = vectorizer.transform([cleaned_input])

        # Предсказание категории всеми моделями
        predictions = [model.predict(input_tfidf)[0] for model in models.values()]

        # Убедимся, что все предсказания - это скалярные значения
        predictions = [pred.item() if isinstance(pred, np.ndarray) else pred for pred in predictions]

        # Выбираем самый частый класс (голосование большинства)
        most_common_pred = Counter(predictions).most_common(1)[0][0]
        predicted_category = label_encoder.inverse_transform([most_common_pred])[0]

        # Вывод предсказанной категории
        st.write(f"### Предсказанный раздел науки: **{predicted_category}**")

        # Генерация ключевых слов
        top_keywords = get_top_keywords(input_tfidf, vectorizer, top_n=5)
        st.write("### Возможные ключевые слова для заголовка (леммы):")
        st.write(", ".join(top_keywords) if top_keywords else "Ключевые слова не найдены.")

        # Поиск похожих статей
        similarities = cosine_similarity(input_tfidf, all_titles_tfidf)[0]
        top_indices = np.argsort(similarities)[::-1][:5]  # Топ-5 по убыванию сходства

        st.write("### Топ-5 похожих статей:")
        found_similar_articles = False
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score > 0:  # Только статьи с ненулевым сходством
                found_similar_articles = True
                title = articles_data.iloc[idx]['title']
                category = label_encoder.inverse_transform([articles_data.iloc[idx]['category']])[0]
                st.write(f"- **{title}** (Раздел: {category}, Сходство: {similarity_score:.4f})")

        if not found_similar_articles:
            st.write("Похожих статей в базе данных не найдено.")

        # Создание облака слов для введенного заголовка
        st.write("### Облако слов для вашего заголовка:")
        create_word_cloud(cleaned_input)

    else:
        st.warning("Пожалуйста, введите заголовок статьи.")

st.markdown("---")
st.markdown("Сделано с ❤️ и с использованием Streamlit и машинного обучения.")