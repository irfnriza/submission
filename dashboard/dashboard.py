import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    all_data = pd.read_csv("main_data.csv")
    
    # Konversi tanggal dan waktu
    all_data["date"] = pd.to_datetime(all_data[["year", "month", "day", "hour"]])
    all_data.set_index("date", inplace=True)

    return all_data

# Load data
all_data = load_data()

# Sidebar untuk memilih visualisasi
st.sidebar.header("Pengaturan Analisis")
analysis_type = st.sidebar.selectbox(
    "Pilih Analisis", 
    ["Tren Polusi Udara", "Heatmap Korelasi", "Pengaruh Arah Angin"]
)

# 1️⃣ **Tren Polusi Udara Harian**
if analysis_type == "Tren Polusi Udara":
    st.subheader("Tren Polusi Udara Harian")

    # Resampling data per hari
    df_daily = all_data.select_dtypes(include=['number']).resample("D").mean()

    # Plot Tren Polusi Udara Harian
    fig, ax = plt.subplots(figsize=(12, 6))
    for polutan in ["PM2.5", "PM10", "NO2", "CO", "O3"]:
        ax.plot(df_daily.index, df_daily[polutan], label=polutan)

    ax.legend()
    ax.set_title("Tren Polusi Udara Harian")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Konsentrasi Polutan")
    plt.xticks(rotation=45)
    
    st.pyplot(fig)

# 2️⃣ **Heatmap Korelasi**
elif analysis_type == "Heatmap Korelasi":
    st.subheader("Heatmap Korelasi antara Variabel")

    # Korelasi antara variabel numerik
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(all_data.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Heatmap Korelasi")

    st.pyplot(fig)

# 3️⃣ **Pengaruh Arah Angin terhadap PM2.5**
elif analysis_type == "Pengaruh Arah Angin":
    st.subheader("Pengaruh Arah Angin terhadap PM2.5")

    # Urutan arah angin
    order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # Barplot rata-rata PM2.5 berdasarkan arah angin
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=all_data["wd"], y=all_data["PM2.5"], order=order, ci=None, palette="coolwarm", ax=ax)
    ax.set_title("Rata-rata PM2.5 Berdasarkan Arah Angin")
    ax.set_xlabel("Arah Angin (wd)")
    ax.set_ylabel("Konsentrasi PM2.5")
    
    st.pyplot(fig)

    # Boxplot distribusi PM2.5 berdasarkan arah angin
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=all_data["wd"], y=all_data["PM2.5"], order=order, palette="coolwarm", ax=ax)
    ax.set_title("Distribusi PM2.5 Berdasarkan Arah Angin")
    ax.set_xlabel("Arah Angin (wd)")
    ax.set_ylabel("PM2.5 (µg/m³)")

    st.pyplot(fig)
