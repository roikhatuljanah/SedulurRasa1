import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
import json
import asyncio
import plotly.graph_objects as go
import plotly.express as px
import random
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

# Fungsi untuk menyimpan hasil tes
def save_test_result(result):
    if not os.path.exists('test_results.json'):
        with open('test_results.json', 'w') as f:
            json.dump([], f)
    
    with open('test_results.json', 'r') as f:
        results = json.load(f)
    
    results.append(result)
    
    with open('test_results.json', 'w') as f:
        json.dump(results, f)

# Fungsi untuk memuat hasil tes
def load_test_results():
    if not os.path.exists('test_results.json'):
        return []
    
    with open('test_results.json', 'r') as f:
        return json.load(f)

# Fungsi untuk menganalisis hasil tes
def analyze_test_results(results):
    categories = {
        "Stres": 0,
        "Depresi": 0,
        "Kecemasan": 0,
        "Burnout": 0
    }
    
    for result in results:
        if result['percentage'] < 25:
            categories["Stres"] += 1
        elif result['percentage'] < 50:
            categories["Depresi"] += 1
        elif result['percentage'] < 75:
            categories["Kecemasan"] += 1
        else:
            categories["Burnout"] += 1
    
    return categories

# Game 1: Tebak Emosi
def emotion_guessing_game():
    st.subheader("Game 1: Tebak Emosi")
    st.write("Tebak emosi yang digambarkan dalam situasi berikut:")

    situations = [
        ("Anda baru saja mendapat promosi di tempat kerja.", "Bahagia"),
        ("Teman Anda membatalkan rencana di menit terakhir.", "Kecewa"),
        ("Anda kehilangan dompet di tempat umum.", "Cemas"),
        ("Seseorang memberi Anda pujian yang tulus.", "Senang"),
        ("Anda melihat berita tentang bencana alam.", "Sedih")
    ]

    if 'game1_score' not in st.session_state:
        st.session_state.game1_score = 0
    if 'game1_question' not in st.session_state:
        st.session_state.game1_question = random.choice(situations)

    situation, correct_emotion = st.session_state.game1_question
    st.write(f"Situasi: {situation}")

    user_guess = st.text_input("Apa emosi yang mungkin dirasakan?", key="game1_input")

    if st.button("Cek Jawaban", key="game1_button"):
        if user_guess.lower() == correct_emotion.lower():
            st.success("Benar!")
            st.session_state.game1_score += 1
        else:
            st.error(f"Maaf, jawaban yang benar adalah {correct_emotion}.")
        st.session_state.game1_question = random.choice(situations)
        st.experimental_rerun()

    st.write(f"Skor Anda: {st.session_state.game1_score}")

# Game 2: Coping Strategies
def coping_strategies_game():
    st.subheader("Game 2: Strategi Coping")
    st.write("Pilih strategi coping yang paling sesuai untuk situasi berikut:")

    situations = [
        ("Anda merasa overwhelmed dengan pekerjaan.", 
         ["Membuat daftar prioritas", "Menonton TV seharian", "Mengabaikan semua tugas"],
         "Membuat daftar prioritas"),
        ("Anda merasa cemas tentang presentasi besok.", 
         ["Berlatih presentasi", "Membatalkan presentasi", "Minum alkohol untuk rileks"],
         "Berlatih presentasi"),
        ("Anda sedang konflik dengan teman.", 
         ["Mengabaikan teman tersebut", "Berbicara dan menyelesaikan masalah", "Membicarakan teman di belakang"],
         "Berbicara dan menyelesaikan masalah")
    ]

    if 'game2_score' not in st.session_state:
        st.session_state.game2_score = 0
    if 'game2_question' not in st.session_state:
        st.session_state.game2_question = random.choice(situations)

    situation, options, correct_strategy = st.session_state.game2_question
    st.write(f"Situasi: {situation}")

    user_choice = st.radio("Pilih strategi terbaik:", options, key="game2_radio")

    if st.button("Cek Jawaban", key="game2_button"):
        if user_choice == correct_strategy:
            st.success("Benar! Itu adalah strategi yang baik.")
            st.session_state.game2_score += 1
        else:
            st.error(f"Maaf, strategi yang lebih baik adalah: {correct_strategy}")
        st.session_state.game2_question = random.choice(situations)
        st.experimental_rerun()

    st.write(f"Skor Anda: {st.session_state.game2_score}")

# Game 3: Mindfulness Exercise
def mindfulness_exercise():
    st.subheader("Game 3: Latihan Mindfulness")
    st.write("Ikuti petunjuk berikut untuk latihan mindfulness singkat:")

    exercises = [
        "Tutup mata Anda dan fokus pada nafas Anda selama 30 detik.",
        "Sebutkan 5 hal yang bisa Anda lihat, 4 yang bisa Anda sentuh, 3 yang bisa Anda dengar, 2 yang bisa Anda cium, dan 1 yang bisa Anda rasakan.",
        "Lakukan peregangan ringan selama 1 menit, fokus pada sensasi di tubuh Anda.",
        "Visualisasikan tempat yang tenang dan damai selama 30 detik.",
        "Ambil 5 napas dalam, hitung sampai 4 saat menghirup dan menghembuskan napas."
    ]

    if 'mindfulness_exercise' not in st.session_state:
        st.session_state.mindfulness_exercise = random.choice(exercises)

    st.write(st.session_state.mindfulness_exercise)

    if st.button("Saya sudah melakukannya", key="game3_button"):
        st.success("Bagus! Latihan mindfulness dapat membantu meredakan stres dan meningkatkan kesadaran diri.")
        st.session_state.mindfulness_exercise = random.choice(exercises)
        st.experimental_rerun()

# Fungsi untuk form aduan
def complaint_form():
    st.header("Form Aduan")
    st.write("Silakan pilih jenis aduan yang ingin Anda sampaikan.")

    complaint_type = st.selectbox(
        "Jenis Aduan",
        ["Pilih jenis aduan", "Aduan Kesehatan", "Aduan Tindak Perundungan", "Aduan Tindak Kekerasan", "Aduan Tindak Pelecehan"]
    )

    if complaint_type != "Pilih jenis aduan":
        st.write(f"Anda memilih: {complaint_type}")
        
        # Form untuk semua jenis aduan
        name = st.text_input("Nama (Opsional, bisa diisi 'Anonim')")
        date = st.date_input("Tanggal Kejadian")
        location = st.text_input("Lokasi Kejadian")
        description = st.text_area("Deskripsi Kejadian", height=150)
        
        # Field tambahan berdasarkan jenis aduan
        if complaint_type == "Aduan Kesehatan":
            health_issue = st.text_input("Masalah Kesehatan yang Dialami")
            symptoms = st.text_area("Gejala yang Dirasakan", height=100)
            medical_history = st.text_area("Riwayat Medis (jika ada)", height=100)
        
        elif complaint_type in ["Aduan Tindak Perundungan", "Aduan Tindak Kekerasan", "Aduan Tindak Pelecehan"]:
            perpetrator = st.text_input("Pelaku (jika diketahui, bisa diisi 'Tidak Diketahui')")
            witnesses = st.text_input("Saksi (jika ada, pisahkan dengan koma)")
            impact = st.text_area("Dampak yang Dirasakan", height=100)
        
        evidence = st.file_uploader("Unggah Bukti (jika ada)", type=["jpg", "png", "pdf"])
        
        if st.button("Kirim Aduan"):
            # Di sini Anda bisa menambahkan logika untuk menyimpan aduan, misalnya ke database
            st.success("Aduan Anda telah diterima. Terima kasih telah melaporkan.")
            
            # Membuat ringkasan aduan
            summary = f"""
            Ringkasan Aduan:
            ----------------
            Jenis Aduan: {complaint_type}
            Nama: {name}
            Tanggal Kejadian: {date}
            Lokasi: {location}
            Deskripsi: {description}
            """
            
            if complaint_type == "Aduan Kesehatan":
                summary += f"""
                Masalah Kesehatan: {health_issue}
                Gejala: {symptoms}
                Riwayat Medis: {medical_history}
                """
            elif complaint_type in ["Aduan Tindak Perundungan", "Aduan Tindak Kekerasan", "Aduan Tindak Pelecehan"]:
                summary += f"""
                Pelaku: {perpetrator}
                Saksi: {witnesses}
                Dampak: {impact}
                """
            
            if evidence:
                summary += f"Bukti telah diunggah: {evidence.name}"
            
            st.text_area("Ringkasan Aduan", summary, height=300)
            
            # Opsi untuk mengunduh ringkasan
            st.download_button(
                label="Unduh Ringkasan Aduan",
                data=summary,
                file_name="ringkasan_aduan.txt",
                mime="text/plain"
            )
            
#Fungsi untuk tes mental   
def comprehensive_mental_health_test():
    st.header("Tes Kesehatan Mental Komprehensif")
    st.write("Silakan jawab 25 pertanyaan berikut dengan jujur. Pilih jawaban yang paling sesuai dengan perasaan dan pengalaman Anda dalam 2 minggu terakhir.")

    questions = [
        "Saya merasa cemas atau tegang.",
        "Saya merasa sedih atau tertekan.",
        "Saya kesulitan tidur atau tidur terlalu banyak.",
        "Saya merasa lelah atau kurang berenergi.",
        "Saya kehilangan minat pada aktivitas yang biasanya saya nikmati.",
        "Saya merasa sulit berkonsentrasi.",
        "Saya merasa gelisah atau tidak bisa diam.",
        "Saya merasa tidak berharga atau bersalah.",
        "Saya memiliki pikiran untuk menyakiti diri sendiri.",
        "Saya merasa mudah tersinggung atau marah.",
        "Saya mengalami perubahan nafsu makan yang signifikan.",
        "Saya merasa sulit mengambil keputusan.",
        "Saya merasa kesepian meskipun dikelilingi orang lain.",
        "Saya merasa khawatir tentang masa depan.",
        "Saya mengalami gejala fisik seperti sakit kepala atau sakit perut tanpa sebab medis yang jelas.",
        "Saya merasa sulit mengendalikan kekhawatiran saya.",
        "Saya merasa tidak memiliki harapan tentang masa depan.",
        "Saya merasa tidak nyaman dalam situasi sosial.",
        "Saya mengalami serangan panik atau kecemasan yang intens.",
        "Saya merasa sulit menikmati hal-hal kecil dalam hidup.",
        "Saya merasa terbebani oleh tanggung jawab sehari-hari.",
        "Saya merasa sulit rileks atau santai.",
        "Saya merasa tidak puas dengan diri sendiri.",
        "Saya merasa sulit menyelesaikan tugas-tugas sehari-hari.",
        "Saya merasa hidup tidak berarti atau tidak memiliki tujuan."
    ]

    options = ["Tidak pernah", "Jarang", "Kadang-kadang", "Sering", "Sangat sering"]
    scores = []

    for q in questions:
        score = st.select_slider(q, options=options, value="Tidak pernah")
        scores.append(options.index(score))

    if st.button("Lihat Hasil"):
        total_score = sum(scores)
        max_score = len(questions) * 4  # 4 adalah skor maksimum untuk setiap pertanyaan

        percentage = (total_score / max_score) * 100

        st.subheader("Hasil Tes Kesehatan Mental")

        if percentage < 20:
            result = "Kesehatan mental Anda tampaknya dalam kondisi baik. Tetap jaga keseimbangan hidup dan rutinitas sehat Anda."
        elif percentage < 40:
            result = "Anda mungkin mengalami tingkat stres atau kecemasan ringan. Pertimbangkan untuk meningkatkan kegiatan self-care dan relaksasi."
        elif percentage < 60:
            result = "Anda mungkin mengalami tingkat stres atau kecemasan sedang. Disarankan untuk berbicara dengan teman, keluarga, atau konselor tentang perasaan Anda."
        elif percentage < 80:
            result = "Anda mungkin mengalami tingkat stres atau kecemasan yang cukup tinggi. Sangat disarankan untuk berkonsultasi dengan profesional kesehatan mental."
        else:
            result = "Hasil tes menunjukkan tingkat stres atau kecemasan yang sangat tinggi. Sangat penting untuk segera mencari bantuan profesional kesehatan mental."

        st.write(result)

        # Visualisasi hasil
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Tingkat Stres/Kecemasan"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 20], 'color': "cyan"},
                    {'range': [20, 40], 'color': "royalblue"},
                    {'range': [40, 60], 'color': "lightgreen"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': percentage}}))

        st.plotly_chart(fig)

        # Analisis per kategori
        categories = {
            "Kecemasan": [0, 6, 15, 18, 21],
            "Depresi": [1, 4, 7, 16, 24],
            "Gangguan Tidur": [2],
            "Kelelahan": [3],
            "Konsentrasi": [5, 11],
            "Harga Diri": [8, 22],
            "Emosi": [9, 19],
            "Nafsu Makan": [10],
            "Sosial": [12, 17],
            "Kecemasan Masa Depan": [13],
            "Gejala Fisik": [14],
            "Kebahagiaan": [20, 23]
        }

        st.subheader("Analisis per Kategori")
        category_scores = {}
        for category, question_indices in categories.items():
            category_score = sum([scores[i] for i in question_indices]) / len(question_indices)
            category_scores[category] = (category_score / 4) * 100  # Normalisasi ke persentase

        category_df = pd.DataFrame(list(category_scores.items()), columns=['Kategori', 'Skor'])
        fig = px.bar(category_df, x='Kategori', y='Skor', title='Skor per Kategori')
        st.plotly_chart(fig)

        # Fungsi untuk menghasilkan laporan yang dapat diunduh
        def generate_report():
            buffer = io.StringIO()
            buffer.write("Hasil Tes Kesehatan Mental Komprehensif\n\n")
            buffer.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            buffer.write(f"Skor Total: {total_score} dari {max_score}\n")
            buffer.write(f"Persentase: {percentage:.2f}%\n\n")
            buffer.write(f"Analisis Umum: {result}\n\n")
            buffer.write("Analisis per Kategori:\n")
            for category, score in category_scores.items():
                buffer.write(f"- {category}: {score:.2f}%\n")
            buffer.write("\nDetail Jawaban:\n")
            for q, s in zip(questions, scores):
                buffer.write(f"- {q}: {options[s]}\n")
            buffer.write("\nCatatan: Hasil tes ini tidak menggantikan diagnosis profesional. ")
            buffer.write("Jika Anda memiliki kekhawatiran tentang kesehatan mental Anda, ")
            buffer.write("silakan konsultasikan dengan profesional kesehatan mental.")
            return buffer.getvalue()

        report = generate_report()
        st.download_button(
            label="Unduh Hasil Tes",
            data=report,
            file_name="hasil_tes_kesehatan_mental_komprehensif.txt",
            mime="text/plain"
        )

# Fungsi dashboard (dari kode sebelumnya)
def dashboard():
    st.header("Dashboard Kesehatan Mental")

    # Memuat dan menganalisis hasil tes
    results = load_test_results()
    data = analyze_test_results(results)

    # Membuat DataFrame dari hasil analisis
    df = pd.DataFrame(list(data.items()), columns=['Kategori', 'Jumlah'])

    col1, col2 = st.columns(2)

    with col1:
        # Membuat grafik batang
        fig1 = px.bar(df, x='Kategori', y='Jumlah', title='Distribusi Kesehatan Mental')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Menambahkan grafik pie untuk variasi
        fig2 = px.pie(df, values='Jumlah', names='Kategori', title='Proporsi Kategori Kesehatan Mental')
        st.plotly_chart(fig2, use_container_width=True)

    # Menambahkan grafik trend waktu
    if results:
        dates = [datetime.strptime(result['timestamp'], "%Y-%m-%d %H:%M:%S").date() for result in results]
        date_counts = pd.Series(dates).value_counts().sort_index()
        fig3 = px.line(x=date_counts.index, y=date_counts.values, title='Trend Penggunaan Tes Mental')
        fig3.update_xaxes(title="Tanggal")
        fig3.update_yaxes(title="Jumlah Tes")
        st.plotly_chart(fig3, use_container_width=True)
        
    # Membuat layout dengan 2 kolom lagi
    col3, col4 = st.columns(2)

    with col3:
        # Menambahkan grafik trend waktu penggunaan
        if results:
            dates = [datetime.strptime(result['timestamp'], "%Y-%m-%d %H:%M:%S").date() for result in results]
            date_counts = pd.Series(dates).value_counts().sort_index()
            fig3 = px.line(x=date_counts.index, y=date_counts.values, title='Trend Penggunaan Tes Mental')
            fig3.update_xaxes(title="Tanggal")
            fig3.update_yaxes(title="Jumlah Tes")
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Menambahkan grafik trend skor kesehatan mental
        if results:
            df_scores = pd.DataFrame([(datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S").date(), r['percentage']) for r in results], columns=['Date', 'Score'])
            df_scores = df_scores.sort_values('Date')
            fig4 = px.line(df_scores, x='Date', y='Score', title='Trend Skor Kesehatan Mental')
            fig4.update_xaxes(title="Tanggal")
            fig4.update_yaxes(title="Skor Kesehatan Mental (%)")
            st.plotly_chart(fig4, use_container_width=True)

    # Menampilkan statistik umum
    st.subheader("Statistik Umum")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric("Total Pengunjung", f"{len(results)}")
    
    if results:
        avg_score = sum(result['percentage'] for result in results) / len(results)
        with col_stat2:
            st.metric("Rata-rata Skor", f"{avg_score:.2f}%")
        
        latest_score = results[-1]['percentage']
        score_change = latest_score - avg_score
        with col_stat3:
            st.metric("Skor Terakhir", f"{latest_score:.2f}%", f"{score_change:+.2f}%")

    # Menampilkan statistik umum
    st.subheader("Statistik Umum")
    st.write(f"Total pengunjung yang telah melakukan tes: {len(results)}")
    if results:
        avg_score = sum(result['percentage'] for result in results) / len(results)
        st.write(f"Rata-rata skor kesehatan mental: {avg_score:.2f}%")

    # Menambahkan tiga game edukasi kesehatan mental
    st.markdown("---")
    st.header("Game Edukasi Kesehatan Mental")
    game_col1, game_col2, game_col3 = st.columns(3)

    with game_col1:
        emotion_guessing_game()

    with game_col2:
        coping_strategies_game()

    with game_col3:
        mindfulness_exercise()

# Fungsi chatbot yang diperbarui
def chatbot():
    st.header("Chatbot Kesehatan Mental SedulurRasa")
    st.write("Silakan ajukan pertanyaan atau ungkapkan perasaan Anda tentang kesehatan mental, dan saya akan mencoba membantu.")

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

# Fungsi utama Streamlit
def main():
    st.set_page_config(layout="wide", page_title="SedulurRasa - Dashboard Kesehatan Mental")
    
    # CSS untuk styling
    st.markdown("""
    <style>
        .reportview-container {
            background-color: #f0f2f6
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
        
        .stButton>button {
            background-color: #7868e6;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            width: 100%;
        }
        
        .stButton>button:hover {
            background-color: #ffd700;
            color: #7868e6;
        }
        
        .top-menu {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background-color: #7868e6;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .top-menu button {
            background-color: transparent;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        
        .top-menu button:hover {
            color: #ffd700;
            text-decoration: none;
        }
    </style>
    </style>
    """, unsafe_allow_html=True)

    st.title("SedulurRasa")
    
    menu = ["Dashboard", "Tes Mental", "Chatbot", "Form Pengaduan"]
    
    # Membuat menu horizontal di bagian atas
    cols = st.columns(len(menu))
    for idx, item in enumerate(menu):
        with cols[idx]:
            if st.button(item):
                st.session_state.choice = item
    
    # Inisialisasi pilihan jika belum ada
    if 'choice' not in st.session_state:
        st.session_state.choice = "Dashboard"
    
    # Search bar
    st.text_input("Search here", "")
    
    # Menampilkan konten berdasarkan pilihan
    if st.session_state.choice == "Dashboard":
        dashboard()
    elif st.session_state.choice == "Tes Mental":
        comprehensive_mental_health_test()
    elif st.session_state.choice == "Chatbot":
        chatbot()
    elif st.session_state.choice == "Form Pengaduan":
        complaint_form()

    st.markdown("---")
    st.write("© 2024 SedulurRasa. Semua hak cipta dilindungi.")

if __name__ == "__main__":
    main()