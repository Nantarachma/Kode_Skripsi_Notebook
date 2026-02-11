import streamlit as st
import pandas as pd
import numpy as np

# ===== Konfigurasi Halaman =====
st.set_page_config(
    page_title="Template Streamlit App",
    page_icon="📊",
    layout="wide",
)

# ===== Sidebar =====
st.sidebar.title("Navigasi")
menu = st.sidebar.selectbox("Pilih Halaman", ["Home", "Data", "Visualisasi", "Model"])

# ===== Halaman Home =====
if menu == "Home":
    st.title("📊 Template Streamlit App")
    st.write("Selamat datang di template aplikasi Streamlit.")
    st.write("Gunakan sidebar untuk navigasi antar halaman.")

# ===== Halaman Data =====
elif menu == "Data":
    st.title("📂 Upload & Tampilkan Data")

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview Data")
        st.dataframe(df.head())

        st.subheader("Informasi Dataset")
        st.write(f"Jumlah baris: {df.shape[0]}")
        st.write(f"Jumlah kolom: {df.shape[1]}")
        st.write("Tipe data:")
        st.write(df.dtypes)
    else:
        st.info("Silakan upload file CSV untuk memulai.")

# ===== Halaman Visualisasi =====
elif menu == "Visualisasi":
    st.title("📈 Visualisasi Data")
    st.write("Tambahkan visualisasi data di sini.")

    # Contoh chart dengan data dummy
    st.subheader("Contoh Line Chart")
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
    st.line_chart(chart_data)

# ===== Halaman Model =====
elif menu == "Model":
    st.title("🤖 Halaman Model")
    st.write("Tambahkan logika model machine learning di sini.")
    st.write("Contoh: load model, prediksi, tampilkan hasil.")
