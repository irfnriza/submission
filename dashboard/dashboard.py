import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap
from statsmodels.tsa.seasonal import seasonal_decompose
import geopandas as gpd

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
    

# Main app
st.title("Dashboard Analisis Kualitas Udara")

# Add tabs
tab1, tab2, tab3 = st.tabs(["Tren Kualitas Udara", "Korelasi & Distribusi", "Peta Sebaaran Polusi"])

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
    

with tab2:
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Heatmap Korelasi Antara Polutan dan Faktor Lingkungan")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Heatmap Korelasi Antara Polutan dan Faktor Lingkungan")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Rata-rata PM2.5 Berdasarkan Arah Angin")
        order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=df["wd"], y=df["PM2.5"], order=order, ci=None, palette="coolwarm", ax=ax)
        ax.set_title("Rata-rata PM2.5 Berdasarkan Arah Angin")
        ax.set_xlabel("Arah Angin")
        ax.set_ylabel("PM2.5")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ---- Row 2: Distribusi PM2.5 di Stasiun & Arah Angin ----
    st.markdown("### Distribusi PM2.5")
    col3, col4 = st.columns([2, 1])

    with col3:
        st.markdown("#### Distribusi PM2.5 di Masing-Masing Stasiun")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='station', y='PM2.5', data=df, palette='coolwarm', ax=ax)
        ax.set_title("Distribusi PM2.5 di Masing-Masing Stasiun")
        ax.set_xlabel("Stasiun")
        ax.set_ylabel("PM2.5")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    with col4:
        st.markdown("#### Distribusi PM2.5 Berdasarkan Arah Angin")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df["wd"], y=df["PM2.5"], order=order, palette="coolwarm", ax=ax)
        ax.set_title("Distribusi PM2.5 Berdasarkan Arah Angin")
        ax.set_xlabel("Arah Angin")
        ax.set_ylabel("PM2.5")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ---- Row 3: Boxplot Perbandingan Polutan ----
    st.markdown("### Perbandingan Konsentrasi Polutan")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']], palette="coolwarm", ax=ax)
    ax.set_title("Boxplot Konsentrasi Polutan")
    ax.set_xlabel("Jenis Polutan")
    ax.set_ylabel("Konsentrasi")
    st.pyplot(fig)

    # ---- Row 4: Histogram Distribusi Polutan ----
    st.markdown("### Distribusi Data Polusi")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
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
        # Section for spatial analysis
    st.markdown("### Analisis Spasial")
    st.info("Bagian ini akan menampilkan visualisasi peta persebaran polusi udara berdasarkan lokasi stasiun.")

    # Load station coordinates
    stations = pd.DataFrame({
        'station': ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong'],
        'lat': [39.982, 40.218, 40.286, 39.929, 39.929, 39.914, 40.313, 39.933, 40.128, 39.886, 39.993, 39.882],
        'lon': [116.417, 116.231, 116.172, 116.417, 116.339, 116.184, 116.637, 116.461, 116.654, 116.407, 116.306, 116.366]
    })

    # Calculate the average PM2.5 values for each station per year
    avg_pm25_yearly = df.groupby(['station', 'year'])['PM2.5'].mean().reset_index()

    # Merge the average PM2.5 values with the station coordinates
    stations_avg_pm25_yearly = pd.merge(stations, avg_pm25_yearly, on='station')

    # Convert to GeoDataFrame
    gdf_stations_avg_pm25_yearly = gpd.GeoDataFrame(stations_avg_pm25_yearly, geometry=gpd.points_from_xy(stations_avg_pm25_yearly.lon, stations_avg_pm25_yearly.lat))

    # Create year selection slider
    year_to_filter = st.slider('Pilih Tahun', min_value=int(gdf_stations_avg_pm25_yearly['year'].min()), 
                            max_value=int(gdf_stations_avg_pm25_yearly['year'].max()), 
                            value=int(gdf_stations_avg_pm25_yearly['year'].min()))

    # Filter data for the selected year
    filtered_data = gdf_stations_avg_pm25_yearly[gdf_stations_avg_pm25_yearly['year'] == year_to_filter]

        # Define the color function for the markers
    def get_color(pm25_value):
        if pm25_value <= 50:
            return 'green'
        elif pm25_value <= 100:
            return 'yellow'
        elif pm25_value <= 150:
            return 'orange'
        else:
            return 'red'

    # Create a base map
    m = folium.Map(location=[39.9042, 116.4074], zoom_start=10)

    # Add markers for each station
    for idx, row in filtered_data.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=row['PM2.5'] / 10,  # Size based on PM2.5 concentration
            popup=f"Station: {row['station']}<br>PM2.5: {row['PM2.5']:.2f}",
            color='blue',
            fill=True,
            fill_color=get_color(row['PM2.5']),
            fill_opacity=0.7
        ).add_to(m)

    # Add heatmap layer
    HeatMap(data=filtered_data[['lat', 'lon', 'PM2.5']].values, 
            radius=15, 
            blur=10, 
            max_zoom=1).add_to(m)

    # Add a legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: 120px; 
        border: 2px solid grey; z-index: 9999; font-size: 14px; background-color: white;
        padding: 10px; border-radius: 5px;">
        <p><strong>PM2.5 Level</strong></p>
        <p><span style="color: green;">&#9679;</span> 0-50 (Baik)</p>
        <p><span style="color: yellow;">&#9679;</span> 51-100 (Sedang)</p>
        <p><span style="color: orange;">&#9679;</span> 101-150 (Tidak Sehat)</p>
        <p><span style="color: red;">&#9679;</span> >150 (Berbahaya)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map as an HTML file
    map_html = "pm25_map.html"
    m.save(map_html)

    # Display the map using Streamlit components
    st.components.v1.html(open(map_html, 'r').read(), height=600)

    # Add a summary of the map
    st.write(f"### Statistik PM2.5 untuk Tahun {year_to_filter}")
    st.write(f"Nilai PM2.5 rata-rata: {filtered_data['PM2.5'].mean():.2f}")
    st.write(f"Nilai PM2.5 tertinggi: {filtered_data['PM2.5'].max():.2f} (Stasiun: {filtered_data.loc[filtered_data['PM2.5'].idxmax()]['station']})")
    st.write(f"Nilai PM2.5 terendah: {filtered_data['PM2.5'].min():.2f} (Stasiun: {filtered_data.loc[filtered_data['PM2.5'].idxmin()]['station']})")
