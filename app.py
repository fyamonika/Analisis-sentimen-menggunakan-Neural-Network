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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud

# Ukuran font & DPI eksplisit supaya grafik tidak berantakan
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": False,
})

st.set_page_config(page_title="Sentimen Kualitas Air", page_icon="💧", layout="wide")

# ----------------------------------------------------------------------------
# Konstanta
# ----------------------------------------------------------------------------
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
    "dr": "dari", "pd": "pada", "mlm": "malam", "pagi2": "pagi", "sm": "sama",
    "makasih": "terima kasih", "terimakasih": "terima kasih", "trims": "terima kasih",
    "thanks": "terima kasih", "thx": "terima kasih", "tq": "terima kasih",
    "padem": "padam", "mati2": "mati", "idup": "hidup", "nyala": "menyala",
    "byarpet": "byarpet", "bgmn": "bagaimana", "gmn": "bagaimana", "gimana": "bagaimana",
    "tlg": "tolong", "plis": "tolong", "pls": "tolong", "mohon2": "mohon",
    "pln": "pln", "pemadamannya": "pemadaman", "listriknya": "listrik",
    # tambahan sesuai kosakata Tweet_Data.csv
    "yaelah": "aduh", "yaa": "ya", "yaaa": "ya", "duh": "aduh", "sumpah": "sungguh",
    "info": "informasi", "update": "pembaruan", "syukurlah": "bersyukur", "hidup": "menyala",
    "deh": "", "sih": "", "dong": "", "nih": "", "tuh": "", "lho": "", "kok": "",
}

KATA_POSITIF = {
    "bersih", "jernih", "segar", "layak", "aman", "sehat", "bagus", "baik",
    "normal", "lancar", "mengalir", "pulih", "membaik", "teratasi",
    "terima kasih", "apresiasi", "sigap", "cepat", "berhasil", "bantuan",
    "tersedia", "melimpah", "berkualitas", "terawat", "memuaskan", "alhamdulillah",
}
KATA_NEGATIF = {
    "tercemar", "keruh", "kotor", "bau", "berbau", "limbah", "beracun",
    "berbahaya", "krisis", "kering", "macet", "rusak", "buruk", "keluhan",
    "kecewa", "parah", "susah", "sulit", "langka", "mahal", "gagal",
    "pencemaran", "sakit", "gatal", "kuning", "asin", "berlumpur",
}
STOPWORD = {
    "yang", "di", "ke", "dan", "dari", "untuk", "dengan", "ini", "itu", "pada",
    "adalah", "atau", "juga", "karena", "akan", "ada", "tidak", "sudah", "saya",
    "kami", "kita", "mereka", "dalam", "oleh", "bisa", "udah", "nya", "aja",
    "yg", "dgn", "utk", "tp", "sih", "lah", "kok", "ya", "kan", "rt", "amp",
    "para", "sebagai", "lagi", "masih", "banget", "gak", "nggak", "ga",
}


# ----------------------------------------------------------------------------
# Helper tampilan
# ----------------------------------------------------------------------------
