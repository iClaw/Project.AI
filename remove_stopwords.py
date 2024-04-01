import nltk
from nltk.corpus import stopwords

# Загрузить список стоп-слов на русском языке
stop_words = set(stopwords.words('russian'))

# Удалить стоп-слова из текста
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Загрузить текст из файла
with open("input.txt", "r") as f:
    text = f.read()

# Удалить стоп-слова из текста
processed_text = remove_stopwords(text)

# Сохранить обработанный текст в файл
with open("output.txt", "w") as f:
    f.write(processed_text)