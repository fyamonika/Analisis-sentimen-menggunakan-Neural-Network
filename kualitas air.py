import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

st.set_page_config(page_title="Sentimen Kualitas Air", page_icon="💧", layout="wide")
st.title("💧 Analisis Sentimen Kualitas Air (Neural Network)")

KELAS = ["positif", "netral", "negatif"]
WARNA = {"positif": "#2e9e5b", "netral": "#7f8c8d", "negatif": "#d64545"}
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}

# ---------------- Langkah 4: kamus kata ----------------
KATA_POSITIF = {
    "aman", "lega", "selamat", "bantuan", "pulih", "jernih", "bersih", "segar",
    "layak", "bening", "sehat", "lancar", "bagus", "mantap", "membaik", "baik",
    "sukses", "berhasil", "puas", "senang", "bahagia", "terima", "kasih",
    "alhamdulillah", "syukurlah", "rekomendasi", "apresiasi", "bangga",
    "nyaman", "membantu", "terjaga", "memenuhi", "tersertifikasi", "inovasi",
}
KATA_NEGATIF = {
    "keruh", "bau", "berbau", "kotor", "tercemar", "pencemaran", "parah",
    "rusak", "bahaya", "berbahaya", "limbah", "gatal", "diare", "buruk",
    "kecewa", "mengecewakan", "kesel", "resah", "krisis", "susah", "bocor",
    "mengeluh", "bakteri", "asam", "asin", "berbusa", "berlumpur", "jelek",
    "mengkhawatirkan", "memprihatinkan", "iritasi", "mati", "sulit", "kesulitan",
}
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "ini", "itu", "dengan", "pada",
    "juga", "ada", "sudah", "akan", "atau", "saya", "kami", "kita", "nya", "rt",
    "tapi", "karena", "kalau", "jadi", "yg", "aja", "dong", "sih", "banget",
    "lagi", "udah", "tidak", "bisa", "mau", "hari", "air",
}


# ---------------- Langkah 3: pembersihan teks ----------------
def bersihkan(teks: str) -> str:
    teks = str(teks)
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)       # URL
    teks = re.sub(r"@\w+", " ", teks)                    # mention
    teks = re.sub(r"#\w+", " ", teks)                    # hashtag
    teks = re.sub(r"\brt\b", " ", teks, flags=re.I)      # RT
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)             # simbol & angka
    teks = teks.lower()                                  # case folding
    teks = re.sub(r"\s+", " ", teks).strip()             # spasi berlebih
    return teks


def beri_label(teks: str) -> str:
    kata = teks.split()
    pos = sum(k in KATA_POSITIF for k in kata)
    neg = sum(k in KATA_NEGATIF for k in kata)
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"


@st.cache_data
def muat_data(file):
    if file is None:
        df = pd.read_csv("tweet_data.csv")
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    df = df.dropna(subset=["full_text"]).drop_duplicates(subset=["full_text"]).reset_index(drop=True)
    df["clean_text"] = df["full_text"].apply(bersihkan)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    df["sentimen"] = df["clean_text"].apply(beri_label)
    return df


@st.cache_resource
def latih_model(clean_texts: tuple, labels: tuple, hidden: int, max_iter: int):
    X_tr, X_te, y_tr, y_te = train_test_split(
        list(clean_texts), list(labels), test_size=0.2, random_state=42, stratify=list(labels))
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xtr = tfidf.fit_transform(X_tr)
    Xte = tfidf.transform(X_te)
    mlp = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
                        max_iter=max_iter, random_state=42)
    mlp.fit(Xtr, y_tr)
    return tfidf, mlp, Xte, y_te, mlp.predict(Xte)


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Pengaturan")
upload = st.sidebar.file_uploader("Upload data (CSV/XLSX)", type=["csv", "xlsx"])
hidden = st.sidebar.slider("Neuron tersembunyi", 10, 300, 100, 10)
max_iter = st.sidebar.slider("Max iterasi", 100, 1000, 300, 50)

try:
    df = muat_data(upload)
except Exception as e:
    st.error(f"Gagal membaca data: {e}. Pastikan ada kolom created_at, username, full_text.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Data", "📊 Distribusi", "☁️ WordCloud", "🧠 Model & Evaluasi", "🔮 Prediksi"])

# ---------------- Data ----------------
with tab1:
    st.subheader("Data setelah pembersihan & pelabelan")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah tweet", len(df))
    c2.metric("Rentang waktu", f"{df['created_at'].astype(str).min()[:10]} s/d {df['created_at'].astype(str).max()[:10]}")
    st.dataframe(df[["created_at", "username", "full_text", "clean_text", "sentimen"]], use_container_width=True)

# ---------------- Distribusi (Langkah 5) ----------------
with tab2:
    st.subheader("Distribusi Sentimen")
    jumlah = df["sentimen"].value_counts().reindex(KELAS, fill_value=0)
    warna = [WARNA[k] for k in KELAS]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.bar(KELAS, jumlah.values, color=warna)
        for i, v in enumerate(jumlah.values):
            ax.text(i, v, str(v), ha="center", va="bottom")
        ax.set_ylabel("Jumlah tweet")
        ax.set_title("Bar Chart")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.pie(jumlah.values, labels=KELAS, colors=warna, autopct="%1.1f%%", startangle=90)
        ax.set_title("Pie Chart")
        st.pyplot(fig)
    st.dataframe(jumlah.rename("jumlah").to_frame().T)

# ---------------- WordCloud (Langkah 6) ----------------
with tab3:
    st.subheader("WordCloud per Sentimen")
    cmap = {"positif": "Greens", "netral": "Greys", "negatif": "Reds"}
    cols = st.columns(3)
    for col, k in zip(cols, KELAS):
        teks = " ".join(df.loc[df["sentimen"] == k, "clean_text"])
        with col:
            st.markdown(f"**{k.capitalize()}** {EMOJI[k]}")
            if teks.strip():
                wc = WordCloud(width=500, height=350, background_color="white",
                               colormap=cmap[k], stopwords=STOPWORDS).generate(teks)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Tidak ada data.")

# ---------------- Model (Langkah 7 & 8) ----------------
with tab4:
    st.subheader("Neural Network (MLP) + TF-IDF")
    if df["sentimen"].value_counts().min() < 2:
        st.warning("Setiap kelas butuh minimal 2 data untuk melatih model.")
        st.stop()
    with st.spinner("Melatih model..."):
        tfidf, mlp, Xte, y_te, y_pred = latih_model(
            tuple(df["clean_text"]), tuple(df["sentimen"]), hidden, max_iter)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", f"{acc:.3f}")
    m2.metric("Presisi", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1-Score", f"{f1:.3f}")
    st.caption(f"Data latih {int(len(df) * 0.8)} | data uji {len(y_te)} (rasio 80:20). Metrik = rata-rata weighted.")

    laporan = pd.DataFrame(classification_report(
        y_te, y_pred, labels=KELAS, output_dict=True, zero_division=0)).T.round(3)
    st.dataframe(laporan, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_te, y_pred, labels=KELAS)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_xticklabels(KELAS)
        ax.set_yticks(range(3)); ax.set_yticklabels(KELAS)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title("Confusion Matrix")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        nilai = [acc, prec, rec, f1]
        ax.bar(["Akurasi", "Presisi", "Recall", "F1"], nilai, color="#3b7dd8")
        ax.set_ylim(0, 1.1)
        for i, v in enumerate(nilai):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
        ax.set_title("Metrik Performa")
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(mlp.loss_curve_)
    ax.set_xlabel("Iterasi"); ax.set_ylabel("Loss"); ax.set_title("Kurva Loss Training")
    st.pyplot(fig)

# ---------------- Prediksi (Langkah 9) ----------------
with tab5:
    st.subheader("Prediksi Sentimen Teks Baru")
    teks_baru = st.text_area("Masukkan teks tentang kualitas air:",
                             "Air PDAM di rumah saya keruh dan bau, sangat mengecewakan")
    if st.button("Prediksi"):
        bersih = bersihkan(teks_baru)
        if not bersih:
            st.warning("Teks kosong setelah dibersihkan.")
        else:
            vek = tfidf.transform([bersih])
            hasil = mlp.predict(vek)[0]
            proba = mlp.predict_proba(vek)[0]
            st.markdown(f"### {EMOJI[hasil]} Sentimen: **{hasil.upper()}**")
            st.write("Teks setelah dibersihkan:", f"`{bersih}`")
            dfp = pd.DataFrame({"sentimen": mlp.classes_, "probabilitas": proba}).set_index("sentimen")
            dfp = dfp.reindex(KELAS).fillna(0)
            dfp.index = [f"{EMOJI[k]} {k}" for k in dfp.index]
            st.bar_chart(dfp)
            st.dataframe(dfp.style.format("{:.2%}"))