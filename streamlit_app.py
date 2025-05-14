import os
import psutil
import time
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import streamlit as st
st.set_page_config(layout="wide")  # <-- harus paling atas

import requests
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from huggingface_hub import snapshot_download

# === Konfigurasi HuggingFace URL ===
HF_EMBEDDING_URL = "https://huggingface.co/datasets/hariririski/rich_movie_embeddings/resolve/main/rich_movie_embeddings.pkl"
HF_PARQUET_URL = "https://huggingface.co/datasets/hariririski/rich_movie_embeddings/resolve/main/imdb_tmdb_Sempurna.parquet"
HF_MODEL_ZIP_URL = "https://huggingface.co/datasets/hariririski/rich_movie_embeddings/resolve/main/multilingual_bert.zip"

DATASET_PATH = "imdb/"
MODEL_PATH = os.path.join(DATASET_PATH, "multilingual_bert/")
BERT_PKL = os.path.join(DATASET_PATH, "rich_movie_embeddings.pkl")
MOVIE_FILE = os.path.join(DATASET_PATH, "imdb_tmdb_Sempurna.parquet")

TMDB_API_KEY = "1ec75235bb4ad6c9a7d6b6b8eac6d44e"
PLACEHOLDER_IMAGE = "https://www.jakartaplayers.org/uploads/1/2/5/5/12551960/9585972.jpg?1453219647"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

os.makedirs(DATASET_PATH, exist_ok=True)

# Fungsi unduh dari HuggingFace

def download_from_huggingface(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        st.error(f"❌ Gagal mengunduh dari HuggingFace: {url}")
        st.stop()

# === Ekstrak dan Siapkan File ===
@st.cache_resource
def prepare_files():
    os.makedirs(MODEL_PATH, exist_ok=True)

    # === Download ZIP dan ekstrak jika model belum ada ===
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        zip_path = os.path.join(DATASET_PATH, "multilingual_bert.zip")
        with st.spinner("📥 Mengunduh model ZIP dari HuggingFace..."):
            download_from_huggingface(HF_MODEL_ZIP_URL, zip_path)
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(MODEL_PATH)
        st.success("✅ Model berhasil diunduh dan diekstrak.")

    if not os.path.exists(BERT_PKL):
        with st.spinner("📦 Mengunduh embedding dari HuggingFace..."):
            download_from_huggingface(HF_EMBEDDING_URL, BERT_PKL)

    if not os.path.exists(MOVIE_FILE):
        with st.spinner("📦 Mengunduh dataset parquet dari HuggingFace..."):
            download_from_huggingface(HF_PARQUET_URL, MOVIE_FILE)

prepare_files()

# === Sidebar Navigasi ===
menu = st.sidebar.radio("Menu Halaman", ("Rekomendasi", "Dashboard", "About"))

# === RAM Usage Sidebar Info ===
def display_ram_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    st.sidebar.markdown(f"🧠 RAM digunakan: `{mem_mb:.2f} MB`")

if "last_ram_check" not in st.session_state:
    st.session_state["last_ram_check"] = time.time()

if time.time() - st.session_state["last_ram_check"] > 5:
    display_ram_usage()
    st.session_state["last_ram_check"] = time.time()


# Bersihkan cache hasil pencarian dan dataset jika berpindah menu
prev_menu = st.session_state.get("prev_menu", None)
if prev_menu != menu:
    st.session_state["prev_menu"] = menu

    # Auto-clear hasil rekomendasi kalau pindah dari menu "Rekomendasi"
    if prev_menu == "Rekomendasi" and "results" in st.session_state:
        del st.session_state["results"]

    # Auto-clear dataset kalau pindah dari Dashboard/About ke menu lain
    if prev_menu in ["Dashboard", "About"] and menu not in ["Dashboard", "About"]:
        if "df_movies" in st.session_state:
            del st.session_state["df_movies"]

# === Load Dataset ===
@st.cache_data
def load_data():
    return pd.read_parquet(MOVIE_FILE)

if menu in ["Rekomendasi", "Dashboard", "About"]:
    if "df_movies" not in st.session_state:
        st.session_state["df_movies"] = load_data()
    df_movies = st.session_state["df_movies"]

# === Load Model & Embedding ===
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_PATH)

@st.cache_data
def load_embeddings():
    with open(BERT_PKL, "rb") as f:
        return np.array(pickle.load(f))

# Model dan embedding dimuat di bagian bawah setelah session_state dicek

# === Cek dan muat model serta embedding dari session_state ===
if "bert_embeddings" not in st.session_state:
    st.session_state["bert_embeddings"] = load_embeddings()
bert_embeddings = st.session_state["bert_embeddings"]

if "model" not in st.session_state:
    st.session_state["model"] = load_model()
model = st.session_state["model"]

# === Scikit-learn Nearest Neighbors Index ===
nn_model = NearestNeighbors(n_neighbors=30, metric="cosine")
nn_model.fit(bert_embeddings)


# === TMDb Info ===
def get_movie_details(imdb_id):
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={TMDB_API_KEY}&external_source=imdb_id"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data.get("movie_results"):
                movie = data["movie_results"][0]
                overview = movie.get("overview", "Sinopsis tidak tersedia")
                poster = movie.get("poster_path")
                return overview, TMDB_IMAGE_BASE_URL + poster if poster else PLACEHOLDER_IMAGE
    except:
        pass
    return "Sinopsis tidak tersedia", PLACEHOLDER_IMAGE

def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='id').translate(text) if text else text
    except:
        return text

def search_bert(query, top_n=10, genre=None, min_year=None, max_year=None, min_rating=None):
    q_embed = model.encode([query], normalize_embeddings=True)
    distances, indices = nn_model.kneighbors(q_embed, n_neighbors=top_n * 3)
    results = []
    for i, idx in enumerate(indices[0]):
        movie = df_movies.iloc[idx]
        if genre and genre not in movie["genres"]: continue
        if min_year and int(movie["startYear"]) < min_year: continue
        if max_year and int(movie["startYear"]) > max_year: continue
        if min_rating and float(movie["averageRating"]) < min_rating: continue
        score = 1 - distances[0][i]
        results.append((movie, score))
        if len(results) >= top_n:
            break
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# === Halaman Detail Film ===
query_params = st.query_params
if "movie_id" in query_params:
    imdb_id = query_params["movie_id"]
    movie = df_movies[df_movies["tconst"] == imdb_id].iloc[0]
    sinopsis, poster_url = get_movie_details(imdb_id)
    st.image(poster_url, width=200)
    st.title(f"{movie['primaryTitle']} ({movie['startYear']})")
    st.markdown(f"**Genre:** {movie['genres']}")
    st.markdown(f"**Pemain:** {movie['actors']}")
    st.markdown(f"**Sutradara:** {movie['directors']}")
    st.markdown(f"**Penulis:** {movie['writers']}")
    st.markdown(f"**Rating:** {movie['averageRating']}")
    st.markdown(f"**Sinopsis:** {translate_text(sinopsis)}")
    if st.button("🔙 Kembali"):
        st.query_params.clear()
    st.stop()

if menu == "Dashboard" and "df_movies" not in st.session_state:
    st.warning("Data belum dimuat.")
    st.stop()

if menu == "Rekomendasi" and "df_movies" not in st.session_state:
    st.warning("Data belum dimuat.")
    st.stop()

if menu == "About" and "df_movies" not in st.session_state:
    st.warning("Data belum dimuat.")
    st.stop()

# === Halaman Rekomendasi ===


if menu == "Rekomendasi":
    st.title("\U0001F3AC Rekomendasi Film IMDb (BERT-Based)")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("\U0001F50E Deskripsi Film")
        query = st.text_area("Masukkan deskripsi film:", height=100)
        selected_genre = st.selectbox("Genre:", ["Semua"] + sorted(set(",".join(df_movies["genres"].dropna()).split(","))))
        min_year, max_year = st.slider("Tahun Rilis:", 2000, 2025, (2000, 2025))
        min_rating = st.slider("Minimal Rating:", 0.0, 10.0, 5.0)

        if st.button("\U0001F50D Cari Rekomendasi") and query:
            # Hapus hasil lama
            st.session_state.pop("results", None)

            # Tampilkan spinner
            with col2:
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown(
                    "\u231B <b>Sedang mencari rekomendasi film terbaik berdasarkan deskripsi...</b>",
                    unsafe_allow_html=True
                )

            # Jalankan pencarian
            results = search_bert(
                query, top_n=10,
                genre=None if selected_genre == "Semua" else selected_genre,
                min_year=min_year, max_year=max_year,
                min_rating=min_rating
            )

            # Simpan hasil baru
            st.session_state["results"] = results

            # Hapus spinner
            with col2:
                spinner_placeholder.empty()

    with col2:
        if "results" in st.session_state:
            st.subheader("\U0001F3AC Hasil:")
            cols = st.columns(5)

            st.markdown("""
                <style>
                .poster-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                    padding: 5px;
                    height: 340px;
                    overflow: hidden;
                }
                .poster-container img {
                    width: 140px;
                    height: 210px;
                    object-fit: cover;
                    border-radius: 8px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                }
                .poster-title {
                    font-size: 13px;
                    font-weight: bold;
                    line-height: 1.2;
                    max-height: 2.6em;
                    overflow-wrap: break-word;
                    white-space: normal; /* ubah dari nowrap */
                    text-overflow: ellipsis;
                    margin-top: 6px;
                }
                .poster-meta {
                    font-size: 11px;
                    line-height: 1.2;
                    max-height: 2.5em;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }
                </style>
            """, unsafe_allow_html=True)

            for i, (movie, score) in enumerate(st.session_state["results"]):
                imdb_id = movie["tconst"]
                title = f"{movie['primaryTitle']} ({movie['startYear']})"
                genres = movie["genres"].replace(",", ", ")
                if len(genres) > 40:
                    genres = genres[:37] + "..."
                poster_url = get_movie_details(imdb_id)[1]

                with cols[i % 5]:
                    safe_title = title.encode('utf-8', 'ignore').decode('utf-8')
                    safe_genres = genres.encode('utf-8', 'ignore').decode('utf-8')

                    st.markdown(f"""
                        <div class="poster-container">
                            <a href="?movie_id={imdb_id}">
                                <img src="{poster_url}">
                            </a>
                            <div class="poster-title">
                                <a href="?movie_id={imdb_id}">{safe_title}</a>
                            </div>
                            <div class="poster-meta">⭐ {movie['averageRating']} | 🔥 {score:.2f}</div>
                            <div class="poster-meta">🎭 {safe_genres}</div>
                        </div>
""", unsafe_allow_html=True)


# === Halaman Dashboard ===
elif menu == "Dashboard":
    st.title("📊 Statistik Dataset Film")

    st.markdown(f"**Jumlah total film dalam dataset:** `{len(df_movies):,}` film")

    st.subheader("Distribusi Genre Terpopuler")
    try:
        genre_series = df_movies["genres"].dropna().astype(str)
        genre_counts = pd.Series(",".join(genre_series).split(",")).value_counts().head(10)
        fig, ax = plt.subplots()
        genre_counts.plot(kind="barh", ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Gagal menampilkan distribusi genre: {e}")

    st.subheader("Distribusi Tahun Rilis")
    try:
        df_movies["startYear"] = pd.to_numeric(df_movies["startYear"], errors="coerce")
        fig2, ax2 = plt.subplots()
        df_movies["startYear"].dropna().astype(int).hist(bins=30, ax=ax2)
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Gagal menampilkan distribusi tahun rilis: {e}")

    st.subheader("Distribusi Rating IMDb")
    try:
        df_movies["averageRating"] = pd.to_numeric(df_movies["averageRating"], errors="coerce")
        fig3, ax3 = plt.subplots()
        df_movies["averageRating"].dropna().hist(bins=20, ax=ax3)
        st.pyplot(fig3)
    except Exception as e:
        st.warning(f"Gagal menampilkan distribusi rating: {e}")

    st.subheader("Rata-rata Rating per Genre")
    try:
        genre_ratings = []
        for genre in genre_counts.index:
            genre_df = df_movies[df_movies["genres"].str.contains(genre, na=False)]
            avg_rating = pd.to_numeric(genre_df["averageRating"], errors="coerce").dropna().mean()
            genre_ratings.append((genre, avg_rating))
        df_genre_rating = pd.DataFrame(genre_ratings, columns=["Genre", "AvgRating"]).sort_values(by="AvgRating", ascending=False)
        fig4, ax4 = plt.subplots()
        df_genre_rating.set_index("Genre").plot(kind="barh", ax=ax4, legend=False)
        st.pyplot(fig4)
    except Exception as e:
        st.warning(f"Gagal menghitung rata-rata rating genre: {e}")

    st.subheader("Top Sutradara")
    try:
        top_directors = df_movies["directors"].dropna().value_counts().head(10)
        fig5, ax5 = plt.subplots()
        top_directors.plot(kind="barh", ax=ax5)
        st.pyplot(fig5)
    except Exception as e:
        st.warning(f"Gagal menampilkan top sutradara: {e}")

    st.subheader("Top Aktor")
    try:
        top_actors = df_movies["actors"].dropna().value_counts().head(10)
        fig6, ax6 = plt.subplots()
        top_actors.plot(kind="barh", ax=ax6)
        st.pyplot(fig6)
    except Exception as e:
        st.warning(f"Gagal menampilkan top aktor: {e}")

# === Halaman About ===
elif menu == "About":
    
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat sebagai bagian dari proyek mata kuliah **Pengembangan Perangkat Lunak dan Manajemen Proyek** pada program studi **Magister Kecerdasan Buatan**.

    **Tujuan:**
    Memberikan rekomendasi film berdasarkan deskripsi menggunakan model embedding BERT multilingual dan pencarian vektor FAISS.

    **Teknologi:** Streamlit, Sentence-Transformers, FAISS, Pandas, Matplotlib

    **Anggota Kelompok:**
    - 2408207010023 – Hariririski
    - 2408207010002 – Danial Alfayyadh Sihombing
    - 2408207010008 – Muhammad Faris Adzkia
    - 2408207010030 – Alkautsar
    - 2408207010022 – Luthfi Fathurahman
    - 2408207010024 – Teuku Nanda Saputra
    """)

# === Tombol Reset Session di Sidebar ===
if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()