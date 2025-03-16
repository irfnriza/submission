import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap
from statsmodels.tsa.seasonal import seasonal_decompose

# Page configuration
st.set_page_config(layout="wide", page_title="Air Pollution Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/main_data.csv")
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["weekday"] = df["date"].dt.dayofweek  # 0 = Senin, 6 = Minggu
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()
    

st.title("Dashboard Analisis Kualitas Udara")

# Tabs untuk navigasi
tab1, tab2, tab3 = st.tabs(["Tren Waktu", "Korelasi", "Peta Stasiun"])

# Main app
st.title("Dashboard Analisis Kualitas Udara")

# Add tabs
tab1, tab2, tab3 = st.tabs(["Tren Kualitas Udara", "Korelasi & Distribusi", "Analisis Lanjutan"])

with tab1:
    st.subheader("Filter Data")
    
    # Move filters from sidebar to tab1
    col1, col2 = st.columns(2)
    
    with col1:
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        selected_date = st.date_input("Pilih tanggal", min_date, min_value=min_date, max_value=max_date)
    
    with col2:
        option = st.selectbox("Pilih Pola Waktu", ["Harian", "Mingguan", "Bulanan", "Tahunan"])
    
    # Filter data based on selected date
    filtered_df = df[df["date"].dt.date == selected_date]
    
    st.subheader("Data Kualitas Udara untuk Tanggal yang Dipilih")
    if filtered_df.empty:
        st.warning(f"Tidak ada data tersedia untuk tanggal {selected_date}")
    else:
        # Create separate histograms for each pollutant
        st.subheader("Histogram Kualitas Udara")
        cols_to_plot = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        
        # Use columns for better layout of multiple charts
        cols = st.columns(2)
        for i, col_name in enumerate(cols_to_plot):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(8, 5))
                if col_name in filtered_df.columns and not filtered_df[col_name].isnull().all():  # Check if there's valid data
                    sns.histplot(filtered_df[col_name].dropna(), kde=True, bins=20, ax=ax)
                    ax.set_title(f"Distribusi {col_name}")
                    st.pyplot(fig)
                else:
                    st.info(f"Tidak ada data valid untuk {col_name} pada tanggal yang dipilih")
    
    st.subheader("Visualisasi Tren Polusi Udara")
    
    # Add date column if not already in the dataframe 
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Add weekday column if not already in the dataframe
    if 'weekday' not in df.columns:
        df['weekday'] = df['date'].dt.dayofweek
    
    # Display different time patterns based on selection
    if option == "Harian":
        # Group by hour to show daily patterns
        daily_df = df.groupby("hour")[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        daily_df.plot(x="hour", marker='o', ax=ax, colormap='tab10')
        ax.set_title("Pola Perubahan Polusi Udara Sepanjang Hari")
        ax.set_xlabel("Jam")
        ax.set_ylabel("Konsentrasi Polutan")
        ax.legend(loc='upper right')
        ax.grid()
        st.pyplot(fig)
    
    elif option == "Mingguan":
        # Group by weekday to show weekly patterns
        weekly_avg = df.groupby('weekday')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        weekly_avg.plot(marker='o', ax=ax, colormap='tab10')
        ax.set_title("Pola Perubahan Polusi Udara Sepanjang Minggu")
        ax.set_xlabel("Hari")
        ax.set_ylabel("Konsentrasi Polutan")
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min'])
        ax.legend(loc='upper right')
        ax.grid()
        st.pyplot(fig)
        
        # Additional boxplot for PM2.5 distribution by weekday
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='weekday', y='PM2.5', data=df, palette='coolwarm', ax=ax)
        ax.set_title("Distribusi PM2.5 Berdasarkan Hari")
        ax.set_xlabel("Hari")
        ax.set_ylabel("Konsentrasi PM2.5")
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min'])
        st.pyplot(fig)
    
    elif option == "Bulanan":
        # Group by month to show monthly patterns
        monthly_avg = df.groupby('month')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        monthly_avg.plot(marker='o', ax=ax, colormap='tab10')
        ax.set_title("Pola Perubahan Polusi Udara Sepanjang Bulan")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Konsentrasi Polutan")
        ax.set_xticks(range(1, 13))
        ax.legend(loc='upper right')
        ax.grid()
        st.pyplot(fig)
        
        # Additional boxplot for PM2.5 distribution by month
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='month', y='PM2.5', data=df, palette='coolwarm', ax=ax)
        ax.set_title("Distribusi PM2.5 per Bulan")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Konsentrasi PM2.5")
        st.pyplot(fig)
    
    else:  # Tahunan
        # Group by year to show yearly patterns
        yearly_avg = df.groupby('year')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        yearly_avg.plot(marker='o', ax=ax, colormap='tab10')
        ax.set_title("Pola Perubahan Polusi Udara Sepanjang Tahun")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Konsentrasi Polutan")
        ax.legend(loc='upper right')
        ax.grid()
        st.pyplot(fig)
        
        # Long-term trend analysis
        # Make a copy of the dataframe to avoid the SettingWithCopyWarning
        df_copy = df.copy()
        if not df_copy.index.equals(df_copy["date"]):
            df_copy.set_index('date', inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df_copy['PM2.5'].rolling(window=30).mean().plot(label='PM2.5 (30 hari rata-rata)', color='r', ax=ax)
        df_copy['NO2'].rolling(window=30).mean().plot(label='NO2 (30 hari rata-rata)', color='b', ax=ax)
        ax.set_title("Tren Jangka Panjang PM2.5 dan NO2")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Konsentrasi")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
    
    # Add boxplot for all pollutants
    st.subheader("Perbandingan Konsentrasi Polutan")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']], palette="coolwarm", ax=ax)
    ax.set_title("Boxplot Konsentrasi Polutan")
    ax.set_xlabel("Jenis Polutan")
    ax.set_ylabel("Konsentrasi")
    st.pyplot(fig)
    
    # Display comprehensive time pattern visualization
    st.subheader("Pola Perubahan Polusi Udara")
    # Calculate time-based averages
    yearly_avg = df.groupby('year')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
    monthly_avg = df.groupby('month')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
    weekly_avg = df.groupby('weekday')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
    hourly_avg = df.groupby('hour')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Yearly pattern
    yearly_avg.plot(marker='o', ax=axes[0, 0], colormap='tab10')
    axes[0, 0].set_title("Pola Perubahan Polusi Udara Sepanjang Tahun")
    axes[0, 0].set_xlabel("Tahun")
    axes[0, 0].set_ylabel("Konsentrasi Polutan")
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid()
    
    # 2. Monthly pattern
    monthly_avg.plot(marker='o', ax=axes[0, 1], colormap='tab10')
    axes[0, 1].set_title("Pola Perubahan Polusi Udara Sepanjang Bulan")
    axes[0, 1].set_xlabel("Bulan")
    axes[0, 1].set_ylabel("Konsentrasi Polutan")
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid()
    
    # 3. Weekly pattern
    weekly_avg.plot(marker='o', ax=axes[1, 0], colormap='tab10')
    axes[1, 0].set_title("Pola Perubahan Polusi Udara Sepanjang Minggu")
    axes[1, 0].set_xlabel("Hari")
    axes[1, 0].set_ylabel("Konsentrasi Polutan")
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min'])
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid()
    
    # 4. Daily pattern
    hourly_avg.plot(marker='o', ax=axes[1, 1], colormap='tab10')
    axes[1, 1].set_title("Pola Perubahan Polusi Udara Sepanjang Hari")
    axes[1, 1].set_xlabel("Jam")
    axes[1, 1].set_ylabel("Konsentrasi Polutan")
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid()
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Analisis Korelasi dan Distribusi PM2.5")
    
    # Row 1: Correlation Heatmap and Wind Direction Average
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Heatmap Korelasi Antara Polutan dan Faktor Lingkungan")
        # Hitung korelasi
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Heatmap Korelasi Antara Polutan dan Faktor Lingkungan")
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretasi Korelasi:**
        * 1.0: Korelasi positif sempurna
        * 0.7 - 0.9: Korelasi positif kuat
        * 0.4 - 0.6: Korelasi positif sedang
        * 0.1 - 0.3: Korelasi positif lemah
        * 0: Tidak ada korelasi
        * -0.1 - -0.3: Korelasi negatif lemah
        * -0.4 - -0.6: Korelasi negatif sedang
        * -0.7 - -0.9: Korelasi negatif kuat
        * -1.0: Korelasi negatif sempurna
        """)
    
with col2:
    st.markdown("### Rata-rata PM2.5 Berdasarkan Arah Angin")
    # Define wind direction order
    order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # Plot rata-rata PM2.5 untuk setiap arah angin
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=df["wd"], y=df["PM2.5"], order=order, ci=None, palette="coolwarm", ax=ax)
    ax.set_title("Rata-rata PM2.5 Berdasarkan Arah Angin")
    ax.set_xlabel("Arah Angin (wd)")
    ax.set_ylabel("Konsentrasi PM2.5")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretasi:**
    * Arah angin dari Timur (E) dan Timur-Tenggara (ESE) menunjukkan konsentrasi PM2.5 tertinggi
    * Arah angin dari Barat-Barat Laut (WNW) dan Barat Laut (NW) menunjukkan konsentrasi PM2.5 terendah
    * Pola ini dapat mengindikasikan lokasi sumber polusi utama
    """)

# Row 2: PM2.5 Distribution by Station and Wind Direction
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Distribusi PM2.5 di Masing-Masing Stasiun")
    # Visualisasi boxplot untuk data polusi di masing-masing stasiun
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='station', y='PM2.5', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Distribusi PM2.5 di Masing-Masing Stasiun")
    ax.set_xlabel("Stasiun")
    ax.set_ylabel("Konsentrasi PM2.5")
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretasi:**
    * Beberapa stasiun menunjukkan variasi yang besar dalam konsentrasi PM2.5
    * Terdapat outlier yang signifikan di beberapa stasiun pengukuran
    * Stasiun di area perkotaan umumnya memiliki konsentrasi PM2.5 lebih tinggi
    """)

with col2:
    st.markdown("### Distribusi PM2.5 Berdasarkan Arah Angin")
    # Boxplot Arah Angin dan Polusi Udara
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=df["wd"], y=df["PM2.5"], order=order, palette="coolwarm", ax=ax)
    ax.set_title("Distribusi PM2.5 Berdasarkan Arah Angin")
    ax.set_xlabel("Arah Angin (wd)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretasi:**
    * Distribusi PM2.5 menunjukkan variasi yang signifikan berdasarkan arah angin
    * Arah angin dari timur (E, ENE, ESE) cenderung membawa tingkat polusi yang lebih tinggi
    * Arah angin dari barat (W, WNW, NW) umumnya lebih bersih
    * Outlier tinggi terlihat pada beberapa arah angin tertentu
    """)

# Additional visualization: Distribution histograms for pollution data
st.markdown("### Distribusi Data Polusi")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
for i, pollutant in enumerate(pollutants):
    if pollutant in df.columns:
        sns.histplot(df[pollutant], bins=50, kde=True, ax=axes[i], color='steelblue', edgecolor='black')
        axes[i].set_title(f'Distribusi {pollutant}')
        axes[i].set_xlabel(f'Konsentrasi {pollutant}')
        axes[i].set_ylabel('Frekuensi')

plt.tight_layout()
st.pyplot(fig)


with tab3:
    st.subheader("Analisis Lanjutan dan Prediksi")
    
    # Section for spatial analysis
    st.markdown("### Analisis Spasial")
    st.info("Bagian ini akan menampilkan visualisasi peta persebaran polusi udara berdasarkan lokasi stasiun.")
    
    # Add a placeholder for future map visualization
    st.image("https://via.placeholder.com/800x400?text=Peta+Distribusi+Polusi+Udara", caption="Placeholder untuk visualisasi peta")
    
    # Advanced correlation analysis
    st.markdown("### Pengaruh Faktor Meteorologi terhadap Polusi")
    
    # Select parameters for analysis
    col1, col2 = st.columns(2)
    with col1:
        param_x = st.selectbox("Pilih Parameter X", ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'])
    with col2:
        param_y = st.selectbox("Pilih Polutan Y", ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'])
    
    # Create scatter plot with regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=param_x, y=param_y, data=df, scatter_kws={"alpha":0.3}, line_kws={"color":"red"}, ax=ax)
    ax.set_title(f"Hubungan antara {param_x} dan {param_y}")
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Calculate correlation coefficient
    correlation = df[param_x].corr(df[param_y])
    st.markdown(f"**Koefisien Korelasi**: {correlation:.4f}")
    
    # Interpretation of correlation
    if abs(correlation) < 0.3:
        st.markdown("Korelasi lemah: Terdapat sedikit atau tidak ada hubungan linear antara kedua variabel.")
    elif abs(correlation) < 0.7:
        st.markdown("Korelasi sedang: Terdapat hubungan moderat antara kedua variabel.")
    else:
        st.markdown("Korelasi kuat: Terdapat hubungan yang signifikan antara kedua variabel.")
    
    # Time series decomposition
    st.markdown("### Dekomposisi Deret Waktu PM2.5")
    
    # Check if we have enough data for decomposition
    if len(df) > 14:  # Need at least 2 periods for seasonal decomposition
        # Resample to daily data for decomposition
        try:
            # Prepare data for decomposition
            df_copy = df.copy()
            if not df_copy.index.equals(df_copy['date']):
                df_copy.set_index('date', inplace=True)
            
            daily_data = df_copy['PM2.5'].resample('D').mean()
            
            # Fill missing values for decomposition
            daily_data = daily_data.fillna(daily_data.mean())
            
            # Try to decompose with a reasonable frequency
            try:
                # Try with 7 for weekly seasonality
                decomposition = seasonal_decompose(daily_data, model='additive', period=7)
                
                # Plot the decomposition
                fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
                decomposition.observed.plot(ax=axes[0], title='Observed')
                decomposition.trend.plot(ax=axes[1], title='Trend')
                decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
                decomposition.resid.plot(ax=axes[3], title='Residuals')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                **Interpretasi Dekomposisi:**
                * **Trend**: Menunjukkan pola jangka panjang dalam data
                * **Seasonality**: Menunjukkan pola berulang dalam data (misalnya pola mingguan)
                * **Residuals**: Fluktuasi acak setelah komponen trend dan seasonality dihilangkan
                """)
            except:
                st.warning("Tidak dapat melakukan dekomposisi deret waktu dengan data yang tersedia.")
        except:
            st.warning("Data tidak mencukupi untuk dekomposisi deret waktu.")
    else:
        st.warning("Data tidak mencukupi untuk analisis dekomposisi deret waktu.")