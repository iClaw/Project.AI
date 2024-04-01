from transformers import pipeline

# Укажите путь к локально установленной модели
model = pipeline("text-generation", model="C:/Users/User/Desktop/proect/Project.AI-main/gpt/")

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

# Создать приглашение для модели
prompt = "Суммируйте следующий диалог: \n\n"
for message in messages:
    prompt += f"{message['sender']}: {message['text']}\n"

# Сгенерировать сводку диалога
summary = model(prompt, max_length=1000)[0]["generated_text"]

# Вывести сводку
print(summary)