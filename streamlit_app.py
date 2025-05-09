import streamlit as st
import os
import requests
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from huggingface_hub import snapshot_download

# === Konfigurasi Google Drive File IDs ===
DRIVE_IDS = {
    "embedding": "1JSe5M7qgCfINDt9sXDxptAsEtjs4ppzD",
    "dataset": "1azl-WiLcQ3bGoRJt_j9f18Ey4OPT2iZT",
}

DATASET_PATH = "imdb/"
MODEL_PATH = os.path.join(DATASET_PATH, "multilingual_bert/")
BERT_PKL = os.path.join(DATASET_PATH, "rich_movie_embeddings.pkl")
MOVIE_FILE = os.path.join(DATASET_PATH, "imdb_tmdb_Sempurna.parquet")

TMDB_API_KEY = "1ec75235bb4ad6c9a7d6b6b8eac6d44e"
PLACEHOLDER_IMAGE = "https://www.jakartaplayers.org/uploads/1/2/5/5/12551960/9585972.jpg?1453219647"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

os.makedirs(DATASET_PATH, exist_ok=True)

# === Unduh File dari Google Drive ===
def download_from_gdrive(file_id, dest_path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, dest_path)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

# === Ekstrak dan Siapkan File ===
@st.cache_resource
def prepare_files():
    # Pastikan direktori siap
    os.makedirs(MODEL_PATH, exist_ok=True)

    # === Unduh model langsung dari HuggingFace ===
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        with st.spinner("📥 Sedang mengunduh model dari HuggingFace (harap tunggu ±1-2 menit)..."):
            snapshot_download(
                repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                local_dir=MODEL_PATH
            )
        st.success("✅ Model berhasil diunduh dan siap digunakan.")

    # === Download file embedding dan dataset ===
    if not os.path.exists(BERT_PKL):
        with st.spinner("📦 Mengunduh embedding..."):
            download_from_gdrive(DRIVE_IDS["embedding"], BERT_PKL)
    if not os.path.exists(MOVIE_FILE):
        with st.spinner("📦 Mengunduh dataset..."):
            download_from_gdrive(DRIVE_IDS["dataset"], MOVIE_FILE)

prepare_files()

# === Load Dataset ===
@st.cache_data
def load_data():
    return pd.read_parquet(MOVIE_FILE)

df_movies = load_data()

# === Load Model & Embedding ===
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_PATH)

@st.cache_data
def load_embeddings():
    with open(BERT_PKL, "rb") as f:
        return np.array(pickle.load(f))

model = load_model()
bert_embeddings = load_embeddings()

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


# === Sidebar Navigasi ===
st.set_page_config(layout="wide")
menu = st.sidebar.radio("Menu Halaman", ("Rekomendasi", "Dashboard", "About"))


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

    # ➕ Tampilkan jumlah total film
    st.markdown(f"**Jumlah total film dalam dataset:** `{len(df_movies):,}` film")

    st.subheader("Distribusi Genre Terpopuler")
    genre_counts = pd.Series(",".join(df_movies["genres"].dropna()).split(",")).value_counts().head(10)
    fig, ax = plt.subplots()
    genre_counts.plot(kind="barh", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribusi Tahun Rilis")
    df_movies["startYear"] = pd.to_numeric(df_movies["startYear"], errors="coerce")
    fig2, ax2 = plt.subplots()
    df_movies["startYear"].dropna().astype(int).hist(bins=30, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Distribusi Rating IMDb")
    fig3, ax3 = plt.subplots()
    df_movies["averageRating"] = pd.to_numeric(df_movies["averageRating"], errors="coerce")
    df_movies["averageRating"].dropna().hist(bins=20, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Rata-rata Rating per Genre")
    genre_ratings = []
    for genre in genre_counts.index:
        genre_df = df_movies[df_movies["genres"].str.contains(genre, na=False)]
        genre_ratings.append((genre, genre_df["averageRating"].mean()))
    df_genre_rating = pd.DataFrame(genre_ratings, columns=["Genre", "AvgRating"]).sort_values(by="AvgRating", ascending=False)
    fig4, ax4 = plt.subplots()
    df_genre_rating.set_index("Genre").plot(kind="barh", ax=ax4, legend=False)
    st.pyplot(fig4)

    st.subheader("Top Sutradara")
    top_directors = df_movies["directors"].value_counts().head(10)
    fig5, ax5 = plt.subplots()
    top_directors.plot(kind="barh", ax=ax5)
    st.pyplot(fig5)

    st.subheader("Top Aktor")
    top_actors = df_movies["actors"].value_counts().head(10)
    fig6, ax6 = plt.subplots()
    top_actors.plot(kind="barh", ax=ax6)
    st.pyplot(fig6)

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