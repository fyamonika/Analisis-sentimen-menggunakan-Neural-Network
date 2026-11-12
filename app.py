import difflib
import io
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud

st.set_page_config(page_title="Sentimen Kualitas Air", page_icon="💧", layout="wide")

st.markdown("""
<style>
.stApp {background:#fff;color:#303443;}
[data-testid="stSidebar"] {background:#f0f2f6;}
[data-testid="stSidebar"] > div:first-child {padding-top:2.2rem;}
.app-title {display:flex;align-items:center;gap:18px;color:#303443;
 font-size:clamp(27px,3vw,42px);font-weight:800;letter-spacing:-1px;margin:18px 0 8px;}
.app-title .water {font-size:43px;}
.source-card {background:#dcefe7;color:#07833f;border-radius:9px;
 padding:18px 16px;font-size:14px;line-height:1.5;margin-top:18px;}
.metric-label {font-size:14px;color:#344054;margin:8px 0 2px;}
.metric-number {font-size:32px;color:#202938;line-height:1.3;}
.date-number {font-size:27px;white-space:nowrap;}
div[data-testid="stTabs"] button[role="tab"] {font-size:14px;}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#f04452;border-bottom-color:#f04452;}
div[data-testid="stDataFrame"] {border:1px solid #e3e6eb;border-radius:9px;}
h1,h2,h3 {color:#303443;}
@media(max-width:900px) {
 .app-title {font-size:27px;flex-wrap:wrap;}
 .date-number {font-size:20px;white-space:normal;}
}
</style>
""", unsafe_allow_html=True)

LABEL_ORDER = ["positif", "netral", "negatif"]
WARNA = {"positif": "#2e9e5b", "netral": "#4a90d9", "negatif": "#d64545"}
CMAP = {"positif": "Greens", "netral": "Blues", "negatif": "Reds"}
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}

ALIAS_KOLOM = {
    "created_at": ["created_at", "created at", "date", "tanggal", "waktu", "time", "timestamp", "datetime"],
    "username": ["username", "user_name", "user", "screen_name", "nama", "author", "name"],
    "full_text": ["full_text", "text", "tweet", "content", "isi", "komentar", "comment", "teks", "isi_tweet"],
}

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
    # kata khusus topik air / PDAM
    "kran": "keran", "krn2": "karena", "ngeluh": "mengeluh", "keluhkan": "mengeluh",
    "kesal": "kesel", "cemar": "tercemar", "bgs": "bagus", "bgus": "bagus", "mantab": "mantap",
    "jlk": "jelek", "alhamdulilah": "alhamdulillah", "syukur": "syukurlah",
    "bnyk": "banyak", "byk": "banyak", "msh": "masih", "lbh": "lebih", "krg": "kurang",
    "knp": "kenapa", "brp": "berapa", "dlm": "dalam", "kpd": "kepada", "sbg": "sebagai",
    "tsb": "tersebut", "dll": "dan lain lain", "dsb": "dan sebagainya", "iya": "ya",
}

KATA_POSITIF = {
    "aman", "lega", "selamat", "bantuan", "pulih", "jernih", "bersih", "segar", "layak", "bening", "sehat",
    "lancar", "bagus", "mantap", "membaik", "baik", "sukses", "berhasil", "puas", "senang", "bahagia",
    "terima kasih", "alhamdulillah", "syukurlah", "rekomendasi", "apresiasi", "bangga", "nyaman",
    "membantu", "terjaga", "memenuhi", "tersertifikasi", "inovasi",
}
KATA_NEGATIF = {
    "keruh", "bau", "berbau", "kotor", "tercemar", "pencemaran", "parah", "rusak", "bahaya", "berbahaya",
    "limbah", "gatal", "diare", "buruk", "kecewa", "mengecewakan", "kesel", "resah", "krisis", "susah",
    "bocor", "mengeluh", "bakteri", "asam", "asin", "berbusa", "berlumpur", "jelek", "mengkhawatirkan",
    "memprihatinkan", "iritasi", "mati", "sulit", "kesulitan",
}
STOPWORD = {
    "yang", "di", "ke", "dan", "dari", "untuk", "dengan", "ini", "itu", "pada",
    "adalah", "atau", "juga", "karena", "akan", "ada", "tidak", "sudah", "saya",
    "kami", "kita", "mereka", "dalam", "oleh", "bisa", "udah", "nya", "aja",
    "yg", "dgn", "utk", "tp", "sih", "lah", "kok", "ya", "kan", "rt", "amp",
    "para", "sebagai", "lagi", "masih", "banget", "gak", "nggak", "ga", "tapi",
    "kalau", "jadi", "dong", "mau", "hari", "air", "sekali", "saja",
}


def clean_text(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\bRT\b", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalisasi_text(text):
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    kata = [KAMUS_SLANG.get(k, k) for k in text.split()]
    return re.sub(r"\s+", " ", " ".join(kata)).strip()


def kata_diganti(sebelum, sesudah):
    a, b = sebelum.split(), sesudah.split()
    hasil = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag != "equal":
            hasil.append(f"{' '.join(a[i1:i2]) or '∅'} → {' '.join(b[j1:j2]) or '∅'}")
    return ", ".join(hasil)


def hitung_kata(teks, kamus):
    return sum(len(re.findall(rf"\b{re.escape(k)}\b", teks)) for k in kamus)


def label_sentimen(teks):
    pos = hitung_kata(teks, KATA_POSITIF)
    neg = hitung_kata(teks, KATA_NEGATIF)
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"


def baca_file(sumber, nama):
    if nama.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(sumber)
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            if hasattr(sumber, "seek"):
                sumber.seek(0)
            return pd.read_csv(sumber, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("File tidak dapat dibaca.")


def rapikan_kolom(data):
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]
    peta = {}
    for target, alias in ALIAS_KOLOM.items():
        for kolom in data.columns:
            if kolom.lower() in alias:
                peta[kolom] = target
                break
    data = data.rename(columns=peta)
    if "full_text" not in data.columns:
        raise ValueError(f"Kolom teks tidak ditemukan. Kolom terbaca: {list(data.columns)}")
    if "username" not in data.columns:
        data["username"] = "tidak_diketahui"
    if "created_at" not in data.columns:
        data["created_at"] = pd.NaT
    data = data[["created_at", "username", "full_text"]].dropna(subset=["full_text"]).copy()
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce")
    return data


@st.cache_data(show_spinner="Membersihkan dan menormalisasi data...")
def proses_data(data, hapus_duplikat):
    hasil = data.copy()
    hasil["clean_text"] = hasil["full_text"].apply(clean_text)
    hasil["normal_text"] = hasil["clean_text"].apply(normalisasi_text)
    hasil = hasil[hasil["normal_text"].str.len() > 0]
    if hapus_duplikat:
        hasil = hasil.drop_duplicates(subset="normal_text")
    hasil = hasil.reset_index(drop=True)
    hasil["sentimen"] = hasil["normal_text"].apply(label_sentimen)
    return hasil


@st.cache_resource(show_spinner="Melatih model Neural Network...")
def latih_model(teks, label, hidden, max_iter):
    X, y = pd.Series(list(teks)), pd.Series(list(label))
    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOPWORD))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    model = MLPClassifier(
        hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
        max_iter=max_iter, random_state=42
    )
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    return {
        "vec": vec, "model": model, "y_test": y_test.tolist(),
        "y_pred": list(y_pred), "n_train": len(X_train),
        "n_test": len(X_test), "n_fitur": Xtr.shape[1]
    }


def tampil_tabel(data, **kwargs):
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def tampil_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    try:
        st.image(buf.getvalue(), width="stretch")
    except TypeError:
        st.image(buf.getvalue(), use_column_width=True)


def prediksi(teks, hasil_model):
    dibersihkan = clean_text(teks)
    bersih = normalisasi_text(dibersihkan)
    model, vec = hasil_model["model"], hasil_model["vec"]
    proba = model.predict_proba(vec.transform([bersih]))[0]
    label = model.classes_[int(np.argmax(proba))]
    return dibersihkan, bersih, label, dict(zip(model.classes_, proba))


# Header
st.markdown(
    '<div class="app-title"><span class="water">💧</span>'
    '<span>Analisis Sentimen Kualitas Air (Neural Network)</span></div>',
    unsafe_allow_html=True
)
st.caption("Pipeline: cleansing → normalisasi → pelabelan rule-based → TF-IDF → Neural Network (MLP)")

# Sidebar sesuai desain referensi
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    st.markdown("**Upload data (CSV/XLSX)**")
    unggahan = st.file_uploader("Pilih dataset", type=["csv", "xlsx", "xls"])
    hapus_duplikat = st.checkbox("Hapus tweet duplikat", value=True)
    hidden = st.slider("Neuron tersembunyi", 10, 300, 100, step=10)
    max_iter = st.slider("Max iterasi", 100, 1000, 300, step=50)
    st.caption("Aktivasi ReLU • Optimizer Adam • Rasio data latih:uji = 80:20")

if unggahan is not None:
    sumber = unggahan.name
    try:
        mentah = rapikan_kolom(baca_file(unggahan, unggahan.name))
    except Exception as exc:
        st.error(f"Gagal membaca dataset: {exc}")
        st.stop()
else:
    kandidat = ["tweet_data.xlsx", "tweet_data.csv", "Tweet_Data.xlsx", "Tweet_Data.csv"]
    file_default = next((f for f in kandidat if os.path.exists(f)), None)
    if file_default is None:
        st.info("Silakan unggah dataset CSV/XLSX melalui sidebar. Kolom teks dapat bernama full_text, text, tweet, atau content.")
        st.stop()
    sumber = file_default
    try:
        mentah = rapikan_kolom(baca_file(file_default, file_default))
    except Exception as exc:
        st.error(f"Gagal membaca {file_default}: {exc}")
        st.stop()

df = proses_data(mentah, hapus_duplikat)
if len(df) < 5:
    st.error("Data terlalu sedikit untuk membagi data latih dan data uji. Unggah minimal 5 baris teks.")
    st.stop()

with st.sidebar:
    st.markdown(
        f'<div class="source-card">Sumber data: <b>{sumber}</b><br><br>'
        f'<b>{len(df):,}</b> tweet dipakai</div>'.replace(",", "."),
        unsafe_allow_html=True
    )

try:
    hasil_model = latih_model(tuple(df["normal_text"]), tuple(df["sentimen"]), hidden, max_iter)
except Exception as exc:
    st.error(f"Model gagal dilatih: {exc}")
    st.stop()

tab_data, tab_norm, tab_dist, tab_wc, tab_eval, tab_pred = st.tabs([
    "📄 Data", "🧹 Normalisasi", "📊 Distribusi",
    "☁️ WordCloud", "🧠 Model & Evaluasi", "🔮 Prediksi"
])

# TAB DATA
with tab_data:
    st.subheader("Data setelah pembersihan, normalisasi & pelabelan")
    tanggal = mentah["created_at"].dropna()
    rentang = f"{tanggal.min():%Y-%m-%d} s/d {tanggal.max():%Y-%m-%d}" if not tanggal.empty else "Tidak tersedia"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-label">Jumlah tweet</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-number">{len(df):,}</div>'.replace(",", "."), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-label">Rentang waktu</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-number date-number">{rentang}</div>', unsafe_allow_html=True)
    tampil_tabel(
        df[["created_at", "username", "full_text", "clean_text", "normal_text", "sentimen"]],
        height=430
    )
    st.download_button(
        "⬇️ Unduh data hasil proses (CSV)",
        df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="data_kualitas_air_berlabel.csv", mime="text/csv"
    )

# TAB NORMALISASI
with tab_norm:
    st.subheader("🧹 Hasil Normalisasi")
    st.caption("Normalisasi memperbaiki kata tidak baku, singkatan, dan huruf berulang.")
    tabel = df[["created_at", "username", "full_text", "clean_text", "normal_text"]].copy()
    tabel["berubah"] = tabel["clean_text"] != tabel["normal_text"]
    tabel["kata_diganti"] = [
        kata_diganti(a, b) for a, b in zip(tabel["clean_text"], tabel["normal_text"])
    ]
    berubah = int(tabel["berubah"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah tweet", len(tabel))
    c2.metric("Tweet yang berubah", berubah)
    c3.metric("Persentase berubah", f"{berubah / max(len(tabel), 1):.1%}")
    st.markdown("#### Contoh sebelum → sesudah")
    contoh = tabel[tabel["berubah"]].head(5)
    if contoh.empty:
        st.info("Tidak ada tweet yang berubah oleh normalisasi.")
    for _, baris in contoh.iterrows():
        kiri, kanan = st.columns(2)
        kiri.markdown("**Sebelum (cleansing)**")
        kiri.write(baris["clean_text"])
        kanan.markdown("**Sesudah (normalisasi)**")
        kanan.write(baris["normal_text"])
        st.caption(f"Kata diganti: {baris['kata_diganti']}")
        st.divider()
    st.markdown("#### Coba normalisasi teks sendiri")
    uji = st.text_input("Ketik kalimat tidak baku:", "Air PDAM keruuuh bgt, bau lagi")
    if uji.strip():
        bersih_uji = clean_text(uji)
        normal_uji = normalisasi_text(bersih_uji)
        st.write("Setelah cleansing:", bersih_uji)
        st.write("Setelah normalisasi:", normal_uji)
        st.write("Kata diganti:", kata_diganti(bersih_uji, normal_uji) or "-")
    hanya_berubah = st.checkbox("Tampilkan hanya tweet yang berubah")
    tampil = tabel[tabel["berubah"]] if hanya_berubah else tabel
    st.markdown("#### Tabel lengkap")
    tampil_tabel(tampil, height=420)
    df_kamus = pd.DataFrame(list(KAMUS_SLANG.items()), columns=["kata_tidak_baku", "kata_baku"])
    with st.expander("📖 Kamus normalisasi yang dipakai"):
        tampil_tabel(df_kamus, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ data_normalisasi.csv",
        tabel.to_csv(index=False).encode("utf-8-sig"),
        file_name="data_normalisasi.csv", mime="text/csv")
    d2.download_button("⬇️ kamus_normalisasi.csv",
        df_kamus.to_csv(index=False).encode("utf-8-sig"),
        file_name="kamus_normalisasi.csv", mime="text/csv")

# TAB DISTRIBUSI
with tab_dist:
    dist = df["sentimen"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
    total = max(int(dist.sum()), 1)
    cols = st.columns(3)
    for col, lab in zip(cols, LABEL_ORDER):
        col.metric(f"{EMOJI[lab]} {lab.capitalize()}", int(dist[lab]))
        col.caption(f"{dist[lab] / total:.2%} dari total")
    kiri, kanan = st.columns(2)
    with kiri:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.bar(dist.index, dist.values, color=[WARNA[x] for x in dist.index])
        ax.bar_label(bars, padding=3)
        ax.set_title("Jumlah Tweet per Sentimen")
        ax.set_ylabel("Jumlah tweet")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        tampil_plot(fig)
    with kanan:
        ada = dist[dist > 0]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.pie(ada.values, labels=[x.capitalize() for x in ada.index],
               colors=[WARNA[x] for x in ada.index], autopct="%1.1f%%",
               startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title("Proporsi Sentimen")
        ax.axis("equal")
        fig.tight_layout()
        tampil_plot(fig)

# TAB WORDCLOUD
with tab_wc:
    cols = st.columns(3)
    for col, lab in zip(cols, LABEL_ORDER):
        with col:
            st.subheader(f"{EMOJI[lab]} {lab.capitalize()}")
            teks = " ".join(df.loc[df["sentimen"] == lab, "normal_text"])
            if teks.strip():
                wc = WordCloud(width=600, height=400, background_color="white",
                               colormap=CMAP[lab], stopwords=STOPWORD,
                               max_words=100).generate(teks)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                fig.tight_layout(pad=0)
                tampil_plot(fig)
            else:
                st.info("Tidak ada data.")

# TAB MODEL & EVALUASI
with tab_eval:
    y_test, y_pred = hasil_model["y_test"], hasil_model["y_pred"]
    labels_ada = [lab for lab in LABEL_ORDER if lab in set(df["sentimen"])]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data latih (80%)", hasil_model["n_train"])
    c2.metric("Data uji (20%)", hasil_model["n_test"])
    c3.metric("Fitur TF-IDF", hasil_model["n_fitur"])
    c4.metric("Neuron hidden", hidden)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    ringkas = pd.DataFrame({
        "Metrik": ["Akurasi", "Presisi (weighted)", "Recall (weighted)", "F1-Score (weighted)"],
        "Nilai": [accuracy_score(y_test, y_pred), p, r, f]
    }).round(4)
    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Tabel metrik")
        tampil_tabel(ringkas, hide_index=True)
        st.subheader("Laporan per kelas")
        laporan = classification_report(
            y_test, y_pred, labels=labels_ada, output_dict=True, zero_division=0
        )
        tampil_tabel(pd.DataFrame(laporan).T.round(3))
    with kanan:
        st.subheader("Grafik metrik")
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        bars = ax.bar(ringkas["Metrik"], ringkas["Nilai"], color="#4a90d9")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        tampil_plot(fig)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=labels_ada)
    _, tengah, _ = st.columns([1, 2, 1])
    with tengah:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels_ada)), labels_ada)
        ax.set_yticks(range(len(labels_ada)), labels_ada)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        for i in range(len(labels_ada)):
            for j in range(len(labels_ada)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        tampil_plot(fig)
    st.warning(
        "Catatan: label dibuat otomatis dari kamus kata positif/negatif, lalu kata yang sama "
        "menjadi fitur TF-IDF. Akurasi tinggi belum membuktikan model memahami sentimen sebenarnya. "
        "Untuk evaluasi kuat, beri label manual pada sebagian tweet."
    )

# TAB PREDIKSI
with tab_pred:
    st.subheader("Prediksi sentimen teks baru")
    teks_baru = st.text_area(
        "Masukkan teks", height=110,
        value="Alhamdulillah air sumur sekarang jernih dan aman diminum, terima kasih petugas"
    )
    if st.button("Prediksi Sentimen", type="primary"):
        if not teks_baru.strip():
            st.warning("Teks masih kosong. Ketik kalimat terlebih dahulu.")
        else:
            dibersihkan, bersih, label, proba = prediksi(teks_baru, hasil_model)
            st.markdown(f"### {EMOJI[label]} {label.upper()}")
            st.write(f"**Setelah cleansing:** {dibersihkan}")
            st.write(f"**Setelah normalisasi:** {bersih}")
            for kelas in LABEL_ORDER:
                if kelas in proba:
                    st.write(f"{EMOJI[kelas]} {kelas.capitalize()} — {proba[kelas]:.2%}")
                    st.progress(float(proba[kelas]))
