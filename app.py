import io
import os
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
FOLDER = os.path.dirname(os.path.abspath(__file__))

# ---------------- Kamus ----------------
KATA_POSITIF = {
    "aman", "lega", "selamat", "bantuan", "pulih", "jernih", "bersih", "segar",
    "layak", "bening", "sehat", "lancar", "bagus", "mantap", "membaik", "baik",
    "sukses", "berhasil", "puas", "senang", "bahagia", "terima kasih",
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
    "lagi", "udah", "tidak", "bisa", "mau", "hari", "air", "sekali", "saja",
}

# Kamus normalisasi: kata tidak baku / singkatan -> kata baku
KAMUS_SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak",
    "engga": "tidak", "tdk": "tidak", "gk": "tidak", "tak": "tidak",
    "udah": "sudah", "udh": "sudah", "sdh": "sudah", "dah": "sudah",
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "untk": "untuk",
    "tp": "tapi", "krn": "karena", "karna": "karena", "krna": "karena",
    "bgt": "sekali", "banget": "sekali", "bngt": "sekali", "jg": "juga", "jga": "juga",
    "sy": "saya", "gue": "saya", "gw": "saya", "aku": "saya", "aja": "saja", "doang": "saja",
    "emg": "memang", "emang": "memang", "blm": "belum", "belom": "belum",
    "lg": "lagi", "lgi": "lagi", "skrg": "sekarang", "skrang": "sekarang",
    "bs": "bisa", "bsa": "bisa", "hrs": "harus", "org": "orang", "orng": "orang",
    "kmrn": "kemarin", "kalo": "kalau", "klo": "kalau", "kl": "kalau",
    "dr": "dari", "pd": "pada", "mlm": "malam", "sm": "sama", "rmh": "rumah",
    "makasih": "terima kasih", "terimakasih": "terima kasih", "trims": "terima kasih",
    "thanks": "terima kasih", "thx": "terima kasih", "tq": "terima kasih", "tks": "terima kasih",
    "bgmn": "bagaimana", "gmn": "bagaimana", "gimana": "bagaimana",
    "tlg": "tolong", "plis": "tolong", "pls": "tolong", "mohon2": "mohon",
    "kran": "keran", "ngeluh": "mengeluh", "keluhkan": "mengeluh",
    "kesal": "kesel", "cemar": "tercemar", "bgs": "bagus", "bgus": "bagus", "mantab": "mantap",
    "jlk": "jelek", "alhamdulilah": "alhamdulillah", "syukur": "syukurlah",
    "bnyk": "banyak", "byk": "banyak", "msh": "masih", "lbh": "lebih", "krg": "kurang",
    "knp": "kenapa", "brp": "berapa", "dlm": "dalam", "kpd": "kepada", "sbg": "sebagai",
    "tsb": "tersebut", "dll": "dan lain lain", "dsb": "dan sebagainya", "iya": "ya",
}

ALIAS_KOLOM = {
    "created_at": ["created_at", "created at", "date", "tanggal", "waktu", "time", "timestamp", "datetime"],
    "username": ["username", "user_name", "user", "screen_name", "nama", "author", "name"],
    "full_text": ["full_text", "text", "tweet", "content", "isi", "komentar", "comment", "teks", "isi_tweet"],
}


# ---------------- Preprocessing ----------------
def bersihkan(teks: str) -> str:
    """Cleansing + case folding."""
    teks = str(teks)
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)       # URL
    teks = re.sub(r"@\w+", " ", teks)                    # mention
    teks = re.sub(r"#\w+", " ", teks)                    # hashtag
    teks = re.sub(r"\brt\b", " ", teks, flags=re.I)      # RT
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)             # simbol & angka
    teks = teks.lower()                                  # case folding
    teks = re.sub(r"\s+", " ", teks).strip()             # spasi berlebih
    return teks


def normalisasi(teks: str) -> str:
    """Huruf berulang (keruuuh -> keruh) dan kata tidak baku -> baku."""
    teks = re.sub(r"(.)\1{2,}", r"\1", teks)
    kata = [KAMUS_SLANG.get(k, k) for k in teks.split()]
    return re.sub(r"\s+", " ", " ".join(kata)).strip()


def preprocess(teks: str) -> str:
    return normalisasi(bersihkan(teks))


def hitung_kata(teks: str, kamus) -> int:
    return sum(len(re.findall(rf"\b{re.escape(k)}\b", teks)) for k in kamus)


def beri_label(teks: str) -> str:
    pos, neg = hitung_kata(teks, KATA_POSITIF), hitung_kata(teks, KATA_NEGATIF)
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"


# ---------------- Memuat data ----------------
def baca_bytes(nama: str, isi: bytes) -> pd.DataFrame:
    if nama.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(isi))
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(isi), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("File tidak dapat dibaca (encoding tidak dikenali).")


def cari_file_lokal():
    for folder in (FOLDER, os.getcwd()):
        for nama in ("tweet_data.csv", "Tweet_Data.csv", "tweet_data.xlsx", "Tweet_Data.xlsx"):
            p = os.path.join(folder, nama)
            if os.path.exists(p):
                return p
    return None


@st.cache_data(show_spinner=False)
def muat_data(nama: str, isi: bytes) -> pd.DataFrame:
    df = baca_bytes(nama, isi)
    df.columns = [str(c).strip() for c in df.columns]
    peta = {}
    for target, alias in ALIAS_KOLOM.items():
        for c in df.columns:
            if c.lower() in alias:
                peta[c] = target
                break
    df = df.rename(columns=peta)
    if "full_text" not in df.columns:
        raise ValueError(f"Kolom isi tweet (full_text) tidak ditemukan. Kolom yang terbaca: {list(df.columns)}")
    if "username" not in df.columns:
        df["username"] = "tidak_diketahui"
    if "created_at" not in df.columns:
        df["created_at"] = pd.NaT
    df = df[["created_at", "username", "full_text"]].dropna(subset=["full_text"]).copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.drop_duplicates(subset=["full_text"]).reset_index(drop=True)
    df["clean_text"] = df["full_text"].apply(bersihkan)        # cleansing
    df["normal_text"] = df["clean_text"].apply(normalisasi)    # normalisasi
    df = df[df["normal_text"].str.len() > 0].drop_duplicates(subset="normal_text").reset_index(drop=True)
    df["sentimen"] = df["normal_text"].apply(beri_label)
    return df


@st.cache_resource(show_spinner=False)
def latih_model(teks: tuple, labels: tuple, hidden: int, max_iter: int):
    y_all = list(labels)
    strat = y_all if pd.Series(y_all).value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        list(teks), y_all, test_size=0.2, random_state=42, stratify=strat)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xtr = tfidf.fit_transform(X_tr)
    Xte = tfidf.transform(X_te)
    mlp = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
                        max_iter=max_iter, random_state=42)
    mlp.fit(Xtr, y_tr)
    return tfidf, mlp, y_te, mlp.predict(Xte), len(X_tr)


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Pengaturan")
upload = st.sidebar.file_uploader("Upload data (CSV/XLSX)", type=["csv", "xlsx"])
hidden = st.sidebar.slider("Neuron tersembunyi", 10, 300, 100, 10)
max_iter = st.sidebar.slider("Max iterasi", 100, 1000, 300, 50)

if upload is not None:
    nama_file, isi_file = upload.name, upload.getvalue()
else:
    lokal = cari_file_lokal()
    if lokal is None:
        st.info("📂 Belum ada data. Upload file **tweet_data.csv** (atau .xlsx) lewat panel di sebelah kiri, "
                f"atau letakkan file tersebut di folder aplikasi ini:\n\n`{FOLDER}`")
        st.stop()
    nama_file = os.path.basename(lokal)
    with open(lokal, "rb") as fh:
        isi_file = fh.read()

try:
    df = muat_data(nama_file, isi_file)
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

if df.empty:
    st.error("Data kosong setelah pembersihan. Periksa isi kolom teks pada file kamu.")
    st.stop()

st.sidebar.success(f"Sumber data: {nama_file}\n\n{len(df)} tweet dipakai")

# ---------------- Simpan data normalisasi ke file ----------------
kolom_simpan = {"clean_text": "teks_cleansing", "normal_text": "teks_normalisasi"}
df_normal = df[["created_at", "username", "full_text", "clean_text", "normal_text"]].rename(columns=kolom_simpan)
df_berlabel = df[["created_at", "username", "full_text", "clean_text", "normal_text", "sentimen"]].rename(columns=kolom_simpan)
df_kamus = pd.DataFrame(list(KAMUS_SLANG.items()), columns=["kata_tidak_baku", "kata_baku"])

try:
    df_normal.to_csv(os.path.join(FOLDER, "data_normalisasi.csv"), index=False, encoding="utf-8-sig")
    df_berlabel.to_csv(os.path.join(FOLDER, "data_normalisasi_berlabel.csv"), index=False, encoding="utf-8-sig")
    df_kamus.to_csv(os.path.join(FOLDER, "kamus_normalisasi.csv"), index=False, encoding="utf-8-sig")
    tersimpan = True
except OSError:
    tersimpan = False  # folder tidak bisa ditulis (mis. di hosting); gunakan tombol download

# ---------------- Latih model (di luar tab agar tab Prediksi selalu bisa dipakai) ----------------
model_siap = df["sentimen"].value_counts().min() >= 2 and df["sentimen"].nunique() >= 2
if model_siap:
    with st.spinner("Melatih model..."):
        tfidf, mlp, y_te, y_pred, n_latih = latih_model(
            tuple(df["normal_text"]), tuple(df["sentimen"]), hidden, max_iter)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📄 Data", "🧹 Normalisasi", "📊 Distribusi", "☁️ WordCloud", "🧠 Model & Evaluasi", "🔮 Prediksi"])

# ---------------- Data ----------------
with tab1:
    st.subheader("Data setelah pembersihan, normalisasi & pelabelan")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah tweet", len(df))
    tgl = df["created_at"].dropna()
    c2.metric("Rentang waktu", f"{tgl.min():%Y-%m-%d} s/d {tgl.max():%Y-%m-%d}" if len(tgl) else "-")
    st.dataframe(df[["created_at", "username", "full_text", "clean_text", "normal_text", "sentimen"]])

# ---------------- Normalisasi ----------------
with tab2:
    st.subheader("Hasil Normalisasi")
    berubah = df[df["clean_text"] != df["normal_text"]]
    st.write(f"**{len(berubah)}** dari **{len(df)}** tweet berubah pada tahap normalisasi.")
    st.dataframe(berubah[["clean_text", "normal_text"]].rename(
        columns={"clean_text": "sebelum normalisasi", "normal_text": "sesudah normalisasi"}))

    if tersimpan:
        st.success(f"✅ Data tersimpan otomatis di folder aplikasi: `{FOLDER}`  \n"
                   "(data_normalisasi.csv, data_normalisasi_berlabel.csv, kamus_normalisasi.csv)")
    else:
        st.warning("Folder aplikasi tidak bisa ditulis, gunakan tombol download di bawah.")

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ data_normalisasi.csv", df_normal.to_csv(index=False, encoding="utf-8-sig"),
                       "data_normalisasi.csv", "text/csv")
    d2.download_button("⬇️ data_normalisasi_berlabel.csv", df_berlabel.to_csv(index=False, encoding="utf-8-sig"),
                       "data_normalisasi_berlabel.csv", "text/csv")
    d3.download_button("⬇️ kamus_normalisasi.csv", df_kamus.to_csv(index=False, encoding="utf-8-sig"),
                       "kamus_normalisasi.csv", "text/csv")

# ---------------- Distribusi ----------------
with tab3:
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
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.pie(jumlah.values, labels=KELAS, colors=warna, autopct="%1.1f%%", startangle=90)
        ax.set_title("Pie Chart")
        st.pyplot(fig)
        plt.close(fig)
    st.dataframe(jumlah.rename("jumlah").to_frame().T)

# ---------------- WordCloud ----------------
with tab4:
    st.subheader("WordCloud per Sentimen")
    cmap = {"positif": "Greens", "netral": "Greys", "negatif": "Reds"}
    cols = st.columns(3)
    for col, k in zip(cols, KELAS):
        teks = " ".join(df.loc[df["sentimen"] == k, "normal_text"])
        with col:
            st.markdown(f"**{k.capitalize()}** {EMOJI[k]}")
            if teks.strip():
                wc = WordCloud(width=500, height=350, background_color="white",
                               colormap=cmap[k], stopwords=STOPWORDS).generate(teks)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Tidak ada data.")

# ---------------- Model ----------------
with tab5:
    st.subheader("Neural Network (MLP) + TF-IDF")
    if not model_siap:
        st.warning("Model belum bisa dilatih: dibutuhkan minimal 2 kelas sentimen dan setiap kelas minimal 2 data.")
    else:
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Akurasi", f"{acc:.3f}")
        m2.metric("Presisi", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-Score", f"{f1:.3f}")
        st.caption(f"Data latih {n_latih} | data uji {len(y_te)} (rasio 80:20). Metrik = rata-rata weighted.")

        laporan = pd.DataFrame(classification_report(
            y_te, y_pred, labels=KELAS, output_dict=True, zero_division=0)).T.round(3)
        st.dataframe(laporan)

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
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots()
            nilai = [acc, prec, rec, f1]
            ax.bar(["Akurasi", "Presisi", "Recall", "F1"], nilai, color="#3b7dd8")
            ax.set_ylim(0, 1.1)
            for i, v in enumerate(nilai):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
            ax.set_title("Metrik Performa")
            st.pyplot(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot(mlp.loss_curve_)
        ax.set_xlabel("Iterasi"); ax.set_ylabel("Loss"); ax.set_title("Kurva Loss Training")
        st.pyplot(fig)
        plt.close(fig)

# ---------------- Prediksi ----------------
with tab6:
    st.subheader("Prediksi Sentimen Teks Baru")
    if not model_siap:
        st.warning("Model belum tersedia, lihat tab Model & Evaluasi.")
    else:
        teks_baru = st.text_area("Masukkan teks tentang kualitas air:",
                                 "Air PDAM di rumah saya keruh dan bau, sangat mengecewakan")
        if st.button("Prediksi"):
            dibersihkan = bersihkan(teks_baru)
            bersih = normalisasi(dibersihkan)
            if not bersih:
                st.warning("Teks kosong setelah dibersihkan.")
            else:
                vek = tfidf.transform([bersih])
                hasil = mlp.predict(vek)[0]
                proba = mlp.predict_proba(vek)[0]
                st.markdown(f"### {EMOJI[hasil]} Sentimen: **{hasil.upper()}**")
                st.write("Setelah cleansing:", f"`{dibersihkan}`")
                st.write("Setelah normalisasi:", f"`{bersih}`")
                dfp = pd.DataFrame({"sentimen": mlp.classes_, "probabilitas": proba}).set_index("sentimen")
                dfp = dfp.reindex(KELAS).fillna(0)
                dfp.index = [f"{EMOJI[k]} {k}" for k in dfp.index]
                st.bar_chart(dfp)
                st.dataframe(dfp.style.format("{:.2%}"))