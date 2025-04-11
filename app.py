import streamlit as st

st.set_page_config(
    page_title="Instacart ML Project",
    page_icon="🛒🥕", # Optional: Add an emoji icon
    layout="wide"
)

# Optional: Add a title or welcome message to this main landing page
st.title("🛒 Instacart Customer Behavior Analysis")
st.sidebar.success("Select an analysis page above.") # Add hint to sidebar
st.markdown("Welcome! Use the sidebar navigation to explore the project.")
# Add any other introductory text or images you want on the main landing pagels
