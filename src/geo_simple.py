import streamlit as st
import pandas as pd

def show_map():
    df = pd.DataFrame({
        "lat": [40.4093, 40.3777, 40.3950, 40.4200, 40.4500, 40.5769],
        "lon": [49.8671, 50.0136, 49.8412, 49.9200, 49.8900, 49.6457],
    })
    st.map(df, zoom=9)
