import streamlit as st

# Set page title, icon, and layout
st.set_page_config(
    page_title="RAILVIGIL",
    page_icon="🚂",
    layout="centered"
)

# Page title
st.write("# Welcome to RAILVIGIL! 🚂")

# Sidebar with instructions
st.sidebar.success("Select a demo from the sidebar.")

# Additional information about RAILVIGIL
st.markdown(
    """
    RAILVIGIL is a platform for monitoring railway stations using CCTV cameras.
    **👈 Select a demo from the sidebar** to explore various features.
    ### Want to learn more?
    - Check out [RAILVIGIL website](https://railvigil.com)
    - Read our [user documentation](https://docs.railvigil.com)
    - Join our [community forum](https://forum.railvigil.com)
    ### Explore more demos
    - Analyze CCTV footage using advanced algorithms
    - Track objects in real-time on the railway platform
"""
)

# Buttons for additional actions
if st.button("Contact Us"):
    st.write("Feel free to reach out to us at contact@railvigil.com")

if st.button("Explore Demos"):
    st.write("Check out our latest demos to see RAILVIGIL in action")

if st.button("Get Started"):
    st.write("Start using RAILVIGIL today and enhance railway station security")
