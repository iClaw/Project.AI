from transformers import pipeline
import requests
requests.packages.urllib3.util.timeout = 1200  # Увеличить таймаут до 5 минут

# Загрузить модель ruGPT-3.5 13B из Hugging Face
model = pipeline("text-generation", model="ai-forever/ruGPT-3.5-13B")

# Загрузить входные данные из файла
with open("input.txt", "r") as f:
    lines = f.readlines()

# Преобразовать каждую строку в словарь с информацией о сообщении
messages = []
for line in lines:
    timestamp, sender, text = line.strip().split("\t")
    messages.append({
        "timestamp": timestamp,
        "sender": sender,
        "text": text,
    })

# Сортировать сообщения по времени отправки
messages.sort(key=lambda x: x["timestamp"])

# Создать приглашение для модели ruGPT-3.5 13B
prompt = "Суммируйте следующий диалог: \n\n"
for message in messages:
    prompt += f"{message['sender']}: {message['text']}\n"

# Сгенерировать сводку диалога
summary = model(prompt, max_length=128)[0]["generated_text"]

# Вывести сводку
print(summary)