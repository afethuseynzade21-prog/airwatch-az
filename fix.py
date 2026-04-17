content = open('app/streamlit_app.py', encoding='utf-8').read()

# Fix 1: broken line in tab2
content = content.replace(
    'st.warning("No trained model found. Run: `python -m src.train`")\nshow_map()("No trained model found. Run: `python -m src.train`")',
    'st.warning("No trained model found. Run: `python -m src.train`")'
)

# Fix 2: add map in tab3
content = content.replace(
    'fmap = build_folium_map(station_readings)\n            import plotly.express as px',
    'from src.geo_simple import show_map\n            show_map()'
)

open('app/streamlit_app.py', 'w', encoding='utf-8').write(content)
print("DONE")