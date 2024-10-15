import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
import asyncio
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from datetime import datetime, timedelta
from rasa.core.agent import Agent
from rasa.shared.utils.io import raise_warning
from rasa.utils.endpoints import EndpointConfig

# Load Rasa agent
model_path = "./models"  # Sesuaikan dengan path model Rasa Anda
agent = Agent.load(model_path)

# Fungsi untuk mendapatkan respons dari Rasa
async def get_rasa_response(user_input):
    try:
        responses = await agent.handle_text(user_input)
        if responses:
            for response in responses:
                if 'text' in response:
                    return response['text']
            return "Maaf, saya tidak mengerti. Bisakah Anda menjelaskan lebih lanjut?"
        else:
            return "Maaf, saya tidak dapat memproses permintaan Anda saat ini. Bisakah Anda coba lagi?"
    except Exception as e:
        print(f"Error in get_rasa_response: {e}")
        return "Maaf, terjadi kesalahan. Bisakah Anda mencoba lagi?"

# Fungsi untuk menjalankan coroutine dalam Streamlit
def run_async(coroutine):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coroutine)

# Fungsi untuk menyimpan riwayat chat
def save_chat_history(chat_history):
    df = pd.DataFrame([(item['role'], item['message']) for item in chat_history], columns=['Role', 'Message'])
    df.to_csv('chat_history.csv', index=False)

# Fungsi untuk memuat riwayat chat
def load_chat_history():
    try:
        df = pd.read_csv('chat_history.csv')
        return [{'role': row['Role'], 'message': row['Message'], 'id': str(uuid.uuid4())} for _, row in df.iterrows()]
    except FileNotFoundError:
        return []

# Fungsi untuk menganalisis sentimen (sederhana)
def analyze_sentiment(text):
    positive_words = set(['bahagia', 'senang', 'gembira', 'positif', 'baik'])
    negative_words = set(['sedih', 'marah', 'kecewa', 'negatif', 'buruk'])
    
    words = set(text.lower().split())
    
    positive_score = len(words.intersection(positive_words))
    negative_score = len(words.intersection(negative_words))
    
    if positive_score > negative_score:
        return 'Positif'
    elif negative_score > positive_score:
        return 'Negatif'
    else:
        return 'Netral'

# Fungsi untuk membuat grafik analisis sentimen
def plot_sentiment_analysis(chat_history):
    sentiments = [analyze_sentiment(item['message']) for item in chat_history if item['role'] == 'User']
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Analisis Sentimen Percakapan')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    st.pyplot(plt)

# Fungsi untuk mereset riwayat chat
def reset_chat_history():
    st.session_state.chat_history = []
    if os.path.exists('chat_history.csv'):
        os.remove('chat_history.csv')

# Fungsi untuk mendapatkan perasaan pengguna
def get_user_feeling():
    user_feeling = st.text_input("Bagaimana perasaan Anda hari ini?", key="user_feeling_input")
    if user_feeling:
        response = run_async(get_rasa_response(user_feeling))
        st.session_state.chat_history.append({'role': 'User', 'message': user_feeling, 'id': str(uuid.uuid4())})
        st.session_state.chat_history.append({'role': 'Bot', 'message': response, 'id': str(uuid.uuid4())})
        save_chat_history(st.session_state.chat_history)
        st.session_state.conversation_stage = 'random_chat'
        st.experimental_rerun()
        
# Fungsi untuk membuat diagram pie sentimen
def plot_sentiment_pie_chart(chat_history):
    sentiments = [analyze_sentiment(item['message']) for item in chat_history if item['role'] == 'User']
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values)])
    fig.update_layout(title='Analisis Sentimen Percakapan')
    st.plotly_chart(fig)

def plot_usage_line_chart(chat_history):
    # Mengasumsikan chat_history memiliki timestamp
    if not chat_history:
        st.write("Belum ada data penggunaan yang cukup untuk membuat grafik.")
        return

    dates = [datetime.strptime(item.get('timestamp', ''), "%Y-%m-%d %H:%M:%S").date() 
             for item in chat_history 
             if 'timestamp' in item and item['timestamp']]
    
    if not dates:
        st.write("Tidak ada data timestamp yang valid untuk membuat grafik penggunaan.")
        return

    date_counts = Counter(dates)
    
    # Mengisi tanggal yang hilang dengan nilai 0
    all_dates = pd.date_range(min(dates), max(dates))
    for date in all_dates:
        if date.date() not in date_counts:
            date_counts[date.date()] = 0

    df = pd.DataFrame(list(date_counts.items()), columns=['Date', 'Count'])
    df = df.sort_values('Date')
    
    fig = px.line(df, x='Date', y='Count', title='Tren Penggunaan Chatbot')
    st.plotly_chart(fig)

# Fungsi untuk membuat diagram batang topik
def plot_topic_bar_chart(chat_history):
    # Ini adalah fungsi sederhana untuk mengekstrak topik dari pesan
    # Anda mungkin ingin menggantinya dengan analisis topik yang lebih canggih
    def extract_topic(message):
        topics = {
            'Kecemasan': ['cemas', 'khawatir', 'takut'],
            'Depresi': ['sedih', 'depresi', 'putus asa'],
            'Stres': ['stres', 'tertekan', 'overwhelmed'],
            'Tidur': ['tidur', 'insomnia', 'lelah'],
            'Relasi': ['hubungan', 'teman', 'keluarga']
        }
        for topic, keywords in topics.items():
            if any(keyword in message.lower() for keyword in keywords):
                return topic
        return 'Lainnya'

    topics = [extract_topic(item['message']) for item in chat_history if item['role'] == 'User']
    topic_counts = Counter(topics)
    
    df = pd.DataFrame(list(topic_counts.items()), columns=['Topic', 'Count'])
    fig = px.bar(df, x='Topic', y='Count', title='Frekuensi Topik Percakapan')
    st.plotly_chart(fig)

# Fungsi utama Streamlit
def main():
    st.title("SedulurRasa")
    st.write("Selamat datang di Chatbot Kesehatan Mental SedulurRasa. Silakan ajukan pertanyaan atau ungkapkan perasaan Anda tentang kesehatan mental, dan saya akan mencoba membantu.")

    # CSS untuk tampilan chat
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background-color: black;
        border-radius: 10px;
    }
    .user-message {
        align-self: flex-end;
        background-color: #DCF8C6;
        color: black;
        padding: 8px;
        border-radius: 10px;
        max-width: 70%;
    }
    .bot-message {
        align-self: flex-start;
        background-color: #FFFFFF;
        color: black;
        padding: 8px;
        border-radius: 10px;
        max-width: 70%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'conversation_stage' not in st.session_state:
        st.session_state.conversation_stage = 'ask_name'

    # Display chat history
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for item in st.session_state.chat_history:
            if item['role'] == "User":
                chat_html += f'<div class="user-message"><strong>Anda : </strong> {item["message"]}</div>'
            else:
                chat_html += f'<div class="bot-message"><strong>SedulurRasa : </strong> {item["message"]}</div>'
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # Conversation flow
    if st.session_state.conversation_stage == 'ask_name':
        name_input = st.text_input("Siapa nama Anda ? ", key="user_name_input")
        if name_input:
            st.session_state.user_name = name_input
            greeting_message = f"Halo, {st.session_state.user_name}! Salam kenal, aku Sedulurmu, siap mendengarkan."
            st.session_state.chat_history.append({'role': 'Bot', 'message': greeting_message, 'id': str(uuid.uuid4())})
            save_chat_history(st.session_state.chat_history)
            st.session_state.conversation_stage = 'ask_feeling'
            st.experimental_rerun()

    elif st.session_state.conversation_stage == 'ask_feeling':
        get_user_feeling()

    elif st.session_state.conversation_stage == 'random_chat':
        user_input = st.text_input("", key="user_input")
        
        if user_input:
            response = run_async(get_rasa_response(user_input))
            st.session_state.chat_history.append({'role': 'User', 'message': user_input, 'id': str(uuid.uuid4())})
            st.session_state.chat_history.append({'role': 'Bot', 'message': response, 'id': str(uuid.uuid4())})
            save_chat_history(st.session_state.chat_history)
            st.experimental_rerun()

        col1, col2 = st.columns([1.5, 0.5])

        with col1:
            if st.button("Reset Riwayat Chat"):
                reset_chat_history()
                st.session_state.conversation_stage = 'ask_name'
                st.session_state.user_name = None
                st.success("Riwayat chat telah dihapus!")
                st.experimental_rerun()

        with col2:
            if st.button("Unduh Riwayat Chat"):
                df = pd.DataFrame([(item['role'], item['message']) for item in st.session_state.chat_history], columns=['Role', 'Message'])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Unduh sebagai CSV",
                    data=csv,
                    file_name="chat_history.csv",
                    mime="text/csv"
                )
        
        st.markdown("<hr>", unsafe_allow_html=True)

        # Tombol untuk menganalisis sentimen
        if st.button("Analisis Sentimen"):
            plot_sentiment_analysis(st.session_state.chat_history)
            plot_sentiment_pie_chart(st.session_state.chat_history)
            plot_usage_line_chart(st.session_state.chat_history)
            plot_topic_bar_chart(st.session_state.chat_history)

    st.markdown("---")
    st.write("Catatan: Chatbot ini hanya memberikan informasi umum dan bukan pengganti konsultasi dengan profesional kesehatan mental. Jika Anda memiliki masalah kesehatan mental yang serius, silakan hubungi profesional kesehatan atau layanan darurat.")

if __name__ == "__main__":
    main()