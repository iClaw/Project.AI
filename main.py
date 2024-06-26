import openai

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

# Создать приглашение для модели ruGPT-3.5
prompt = "Суммируйте следующий диалог: \n\n"
for message in messages:
    prompt += f"{message['sender']}: {message['text']}\n"

# Вызвать API ruGPT-3.5 для генерации сводки
response = openai.Completion.create(
    model="text-bison-001",
    prompt=prompt,
    temperature=0.7,
)

# Извлечь сгенерированную сводку
summary = response["choices"][0]["text"]

# Вывести сводку
print(summary)