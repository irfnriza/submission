import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap

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

# Sidebar untuk filter
st.sidebar.header("Filter")
min_date = df["date"].min().date()
max_date = df["date"].max().date()
selected_date = st.sidebar.date_input("Pilih tanggal", min_date, min_value=min_date, max_value=max_date)
option = st.sidebar.selectbox("Pilih Pola Waktu", ["Harian", "Mingguan", "Bulanan", "Tahunan"])

# Convert selected_date to datetime for comparison
selected_date_dt = pd.to_datetime(selected_date)
filtered_df = df[df["date"].dt.date == selected_date_dt.date()]

# Tabs untuk navigasi
tab1, tab2, tab3 = st.tabs(["Tren Waktu", "Korelasi", "Peta Stasiun"])

with tab1:
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
    
    if option == "Harian":
        # Group by hour to show daily patterns
        daily_df = df.groupby("hour")["PM2.5"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="hour", y="PM2.5", data=daily_df, ax=ax)
        ax.set_title("Pola Harian PM2.5")
        ax.set_xlabel("Jam")
        ax.set_ylabel("PM2.5 (μg/m³)")
        st.pyplot(fig)
    elif option == "Mingguan":
        # Group by weekday to show weekly patterns
        weekly_df = df.groupby("weekday")["PM2.5"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="weekday", y="PM2.5", data=weekly_df, ax=ax)
        ax.set_title("Pola Mingguan PM2.5")
        ax.set_xlabel("Hari (0=Senin, 6=Minggu)")
        ax.set_ylabel("PM2.5 (μg/m³)")
        st.pyplot(fig)
    elif option == "Bulanan":
        # Group by month to show monthly patterns
        monthly_df = df.groupby("month")["PM2.5"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="month", y="PM2.5", data=monthly_df, ax=ax)
        ax.set_title("Pola Bulanan PM2.5")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("PM2.5 (μg/m³)")
        st.pyplot(fig)
    else:
        # Group by year to show yearly patterns
        yearly_df = df.groupby("year")["PM2.5"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="year", y="PM2.5", data=yearly_df, ax=ax)
        ax.set_title("Pola Tahunan PM2.5")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("PM2.5 (μg/m³)")
        st.pyplot(fig)
    
with tab2:
    st.subheader("Korelasi Antar Parameter")
    # Check if there are at least 2 rows for correlation
    if len(df) > 1:
        pollution_params = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        # Check which parameters exist in the dataframe
        available_params = [param for param in pollution_params if param in df.columns]
        
        if len(available_params) > 1:
            corr_matrix = df[available_params].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", mask=mask, vmin=-1, vmax=1, 
                        annot_kws={"size": 8}, ax=ax)
            ax.set_title("Korelasi Antar Parameter Polusi dan Cuaca")
            st.pyplot(fig)
            
            # Add explanation for correlation
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
        else:
            st.warning("Tidak cukup parameter tersedia untuk analisis korelasi")
    else:
        st.warning("Tidak cukup data untuk analisis korelasi")

# Just the map section code (tab3)
with tab3:
    st.subheader("Peta Stasiun")
    
    try:
        # Load station coordinates
        stations = pd.DataFrame({
            'station': ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 
                        'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong'],
            'lat': [39.982, 40.218, 40.286, 39.929, 39.929, 39.914, 
                    40.313, 39.933, 40.128, 39.886, 39.993, 39.882],
            'lon': [116.417, 116.231, 116.172, 116.417, 116.339, 116.184, 
                    116.637, 116.461, 116.654, 116.407, 116.306, 116.366]
        })

        # Calculate the average PM2.5 values for each station per year - simplified approach
        station_data = []
        for station_name in stations['station'].unique():
            station_df = df[df['station'] == station_name]
            if not station_df.empty:
                years = station_df['year'].unique()
                for year in years:
                    year_data = station_df[station_df['year'] == year]
                    avg_pm25 = year_data['PM2.5'].mean()
                    if pd.notna(avg_pm25):  # Check if value is not NaN
                        station_data.append({
                            'station': station_name,
                            'year': year,
                            'PM2.5': avg_pm25
                        })

        # Create a new DataFrame from the collected data
        avg_pm25_yearly = pd.DataFrame(station_data)
        
        # Create a base map
        m = folium.Map(location=[39.9042, 116.4074], zoom_start=10)

        # Simple approach: Just add markers for each station with their average PM2.5 level
        # Merge station coordinates with PM2.5 data
        if not avg_pm25_yearly.empty:
            stations_with_data = pd.merge(stations, avg_pm25_yearly, on='station')
            
            # Create separate feature groups for each year
            years = sorted(stations_with_data['year'].unique())
            
            for year in years:
                year_data = stations_with_data[stations_with_data['year'] == year]
                year_group = folium.FeatureGroup(name=f"Stasiun PM2.5 {year}")
                
                for _, row in year_data.iterrows():
                    # Color-code markers based on PM2.5 value
                    pm25 = row['PM2.5']
                    if pm25 < 50:
                        color = 'green'
                    elif pm25 < 100:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    # Create popup with info
                    popup_text = f"""
                    <strong>{row['station']}</strong><br>
                    Tahun: {row['year']}<br>
                    PM2.5: {pm25:.2f} μg/m³
                    """
                    
                    # Add marker
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=10,
                        popup=folium.Popup(popup_text, max_width=200),
                        tooltip=f"{row['station']} ({row['year']})",
                        color=color,
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(year_group)
                
                year_group.add_to(m)
        else:
            # If no PM2.5 data available, just show station locations
            for _, row in stations.iterrows():
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=row['station'],
                    tooltip=row['station']
                ).add_to(m)
            
            st.warning("Tidak ada data PM2.5 tersedia untuk divisualisasikan pada peta")

        # Add layer control to toggle between years
        folium.LayerControl().add_to(m)

        # Display the map in Streamlit
        folium_static(m)
        
        # Add a note about the map markers
        st.info("Peta menunjukkan lokasi stasiun pengukuran kualitas udara. Marker berwarna menunjukkan tingkat konsentrasi PM2.5 (hijau = baik, oranye = sedang, merah = buruk).")
        
    except Exception as e:
        st.error(f"Error saat membuat peta: {e}")
        st.info("Pastikan library folium dan streamlit-folium sudah terinstall dengan benar.")
