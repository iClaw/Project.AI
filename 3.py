import os
from transformers import pipeline

# Загрузите текст из файла
with open("C:\\Users\\User\\Desktop\\proect\\Project.AI-main\\text.txt", "r") as f:
    text = f.read()

# Создайте конвейер для суммирования текста
summarizer = pipeline("summarization")

# Суммируйте текст
summary = summarizer(text, max_length=128)  # Ограничение длины резюме до 128 символов

# Выведите резюме
print(summary[0]["summary_text"])