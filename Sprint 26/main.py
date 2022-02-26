import re
import pandas as pd
import numpy as np
import tensorflow as tf
from vncorenlp import VnCoreNLP
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def text_normalization(text):
    text = text.lower().strip()
    return re.sub('[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n]', ' ', text)

if __name__ == '__main__':

    input_text = 'Khoa học và công nghệ'

    input_text = text_normalization(input_text)
    rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
    sentences = rdrsegmenter.tokenize(input_text)

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    tokens = [tokenizer.encode(sentences[0])]
    tokens = pad_sequences(tokens, maxlen=1000, truncating="post", padding="post")

    model = load_model('model')

    y_pred = model.predict(tokens)
    y_cat = np.argmax(y_pred)
    cat = [
        'Chính trị xã hội',
        'Đời sống',
        'Khoa học',
        'Kinh doanh',
        'Pháp luật',
        'Sức khỏe',
        'Thế giới',
        'Thể thao',
        'Văn hóa',
        'Vi tính'
    ]
    print(f'Đề tài của văn bản là {cat[y_cat]}')
