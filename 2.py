import numpy
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from transformers import T5ForConditionalGeneration

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text

def summarize(document_path):
    with open(document_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = preprocess_text(text)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    tokens = tokenizer(text, return_tensors="pt")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    summary_ids = model.generate(tokens["input_ids"], num_beams=4, max_length=128)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    print(summary)

# Call the function summarize
summarize('C:\\Users\\User\\Desktop\\proect\\Project.AI-main\\text.txt')