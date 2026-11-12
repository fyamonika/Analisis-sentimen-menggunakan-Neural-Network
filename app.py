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
