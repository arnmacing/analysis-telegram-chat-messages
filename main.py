import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pymorphy2
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from telethon.sync import TelegramClient
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')

api_id = ''  # заполните это поле
api_hash = ''  # заполните это поле
session_name = 'test_session'

# Cоздание клиента TelegramClient
client = TelegramClient(session_name, api_id, api_hash, system_version="4.16.30-vxCUSTOM")


# Получает все сообщения из указанного чата и добавляет их дату, отправителя и текст в список
async def get_all_messages(chat_username):
    chat_entity = await client.get_entity(chat_username)
    all_messages = []

    async for message in client.iter_messages(chat_entity, reverse=True):
        all_messages.append({
            'date': message.date,
            'sender': message.sender_id,
            'text': message.text
        })

    return all_messages


# Построения графика динамики сообщений по месяцам
def plot_message_dynamics(messages):
    date_counts = {}
    for message in messages:
        message_date = message['date'].date()
        if message_date in date_counts:
            date_counts[message_date] += 1
        else:
            date_counts[message_date] = 1

    years = np.unique([date.year for date in date_counts])
    month_counts = {year: {month: 0 for month in range(1, 13)} for year in years}

    for message_date, count in date_counts.items():
        year = message_date.year
        month = message_date.month
        month_counts[year][month] += count

    months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

    plt.figure(figsize=(12, 8))

    for year in years:
        counts = [month_counts[year].get(month, 0) for month in range(1, 13)]
        plt.plot(months, counts, label=str(year))

    for month in [2, 5, 8, 11]:
        plt.axvline(month - 0.5, color='gray', linestyle='dashed')

    plt.xlabel('Месяц')
    plt.ylabel('Количество сообщений')
    plt.title('Динамика сообщений по месяцам')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Анализ активности по часам дня и построения соответствующего графика
def analyze_hourly_activity(messages):
    hourly_counts = {}

    for message in messages:
        message_date = message['date'].astimezone().replace(second=0, microsecond=0)
        hour = message_date.hour

        if hour in hourly_counts:
            hourly_counts[hour] += 1
        else:
            hourly_counts[hour] = 1

    # Сортировка по часам
    sorted_hourly_counts = sorted(hourly_counts.items(), key=lambda x: x[0])

    hours = [hour for hour, count in sorted_hourly_counts]
    counts = [count for hour, count in sorted_hourly_counts]

    plt.figure(figsize=(10, 6))
    plt.plot(hours, counts, marker='o')
    plt.xlabel('Часы')
    plt.ylabel('Количество сообщений')
    plt.title('Активность по часам дня')
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Очиcтка текста от различных видов нежелательных элементов
def clean_text(text):
    if text is None:
        return ''

    text = re.sub(r'<.*?>', '', text)  # Удаление тегов html
    text = re.sub(r'http\S+', '', text)  # Удаление url
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)  # Удаление непонятных символов

    return text


def clear_message_punctuation(message):
    clean_text = ""
    string_punctuation = string.punctuation + "—" + "«" + "»" + "1" + "2" + "3" + "4" + "5" + "6" + "7" + "8" + "9" + "0"

    for text in message:
        for ch in string_punctuation:
            text = text.replace(ch, "")
        clean_text += text + " "

    return clean_text.strip()


# Обработка текста сообщений, включая удаление пунктуации и лемматизацию слов
def preprocess_text(messages):
    morph = pymorphy2.MorphAnalyzer()

    for message in messages:
        message_text = message['text']
        cleaned_text = clean_text(message_text)
        punctuation_cleaned_text = clear_message_punctuation([cleaned_text])
        word_tokens = word_tokenize(punctuation_cleaned_text)
        lemmatized_text = [morph.parse(word)[0].normal_form for word in word_tokens]
        message['processed_text'] = lemmatized_text

    return messages


# Создание облака слов на основе предварительно обработанных сообщений
def generate_word_cloud(messages):
    preprocessed_messages = preprocess_text(messages)

    text = ' '.join(' '.join(message['processed_text']) for message in preprocessed_messages if message['text'] is not None)
    text = text.lower()

    stop_words = set(stopwords.words('russian'))
    word_tokens = word_tokenize(text)

    filtered_text = [word for word in word_tokens if word not in stop_words]
    cleaned_text = ' '.join(filtered_text)

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(cleaned_text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with client:
        chat_username = 'jaicp'
        all_messages = client.loop.run_until_complete(get_all_messages(chat_username))

        df = pd.DataFrame(all_messages)
        df.to_csv('chat_data.csv', index=False)

        plot_message_dynamics(all_messages)

        analyze_hourly_activity(all_messages)

        generate_word_cloud(all_messages)
