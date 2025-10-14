import streamlit as st
import requests

# Title of the Streamlit app
st.title("MCP OpenMRS Application")

# Input for natural language query
query = st.text_input("Enter your natural language question about the OpenMRS database:")

if st.button("Submit"):
    if query:
        # Send the query to the Flask backend for processing
        response = requests.post("http://localhost:5000/query", json={"query": query})
        
        if response.status_code == 200:
            # Display the response from the backend
            st.success("Response from OpenMRS:")
            st.write(response.json().get("answer", "No answer found."))
        else:
            st.error("Error processing the query. Please try again.")
    else:
        st.warning("Please enter a query before submitting.")