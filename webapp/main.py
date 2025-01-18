import streamlit as st

st.write("Hello World this is Streamlit")
st.button("Reset", type="primary")
if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')