import streamlit as st
from pathlib import Path
import sys
import os

# ----------------------------------------
# ⛓️ Setup Path for Modules
# ----------------------------------------
base_dir = Path(__file__).parent
utils_dir = os.path.join(base_dir, "utils")
pages_dir = os.path.join(base_dir, "pages_streamlit")

sys.path.append(utils_dir)
sys.path.append(pages_dir)

from data_utils import DataLoader
from preprocessing import DataPreprocessor
from label_generation import LabelGenerator

from pages_streamlit import (
    home_pg,
    data_exploration_pg,
    eda_with_climate_event_pg,
    feature_engineering_pg,
    model_training_pg,
    prediction_pg,
    about_pg,
)

# ----------------------------------------
# 🧭 Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Extreme Weather Event Detection and Prediction",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------
# 🧪 Header
# ----------------------------------------
st.title("🌩️ Extreme Weather Event Detection and Prediction - Nepal")
st.markdown("##### Powered by Omdena x NIC Nepal")

# ----------------------------------------
# 📦 Data Loading
# ----------------------------------------
loader = DataLoader()

try:
    loader.load_shape_file()
    loader.load_climate_data()
    st.success("✅ Data files loaded successfully.")
except FileNotFoundError as e:
    st.error(f"❌ {str(e)}")

# ----------------------------------------
# 🧹 Preprocessing
# ----------------------------------------
preprocessor = DataPreprocessor(loader.climate_df, loader.district_shp)
preprocessor.preprocess()

# ----------------------------------------
# 🏷️ Label Generation
# ----------------------------------------
label_generator = LabelGenerator(preprocessor.df)
label_generator.label_generation_pipeline()

# ----------------------------------------
# 🚀 Sidebar Navigation (Dropdown Menu)
# ----------------------------------------
with st.sidebar:
    st.title("🧭 Navigation")
    st.info("Please select a section to explore:")
    page = st.selectbox(
        "Select a Page",
        options=[
            "Home",
            "Data Exploration",
            "EDA with Climate Events",
            "Feature Engineering",
            "Model Training and Evaluation",
            "Prediction",
            "About",
        ],
        index=0,
    )

# ----------------------------------------
# 🧠 Route to Appropriate Page
# ----------------------------------------
if page == "Home":
    home_pg.show()

elif page == "Data Exploration":
    data_exploration_pg.show(gdf=preprocessor.gdf, df=preprocessor.df)

elif page == "EDA with Climate Events":
    eda_with_climate_event_pg.show(
        gdf=preprocessor.gdf,
        df=label_generator.df,
        thresholds=label_generator.thresholds
    )

elif page == "Feature Engineering":
    feature_engineering_pg.show(label_generator.df)

elif page == "Model Training and Evaluation":
    model_training_pg.show()

elif page == "Prediction":
    prediction_pg.show()

elif page == "About":
    about_pg.show()

# ----------------------------------------
# 📎 Footer
# ----------------------------------------
st.markdown("---")
st.markdown("© **Utsab Bhattarai** | _Omdena - NIC Nepal_")
st.caption("This project is a collaborative effort by Omdena and NIC Capacity Building Batch II - Nepal.")
