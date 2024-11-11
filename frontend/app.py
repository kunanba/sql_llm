# app.py

import streamlit as st
import requests
import uuid

# Set the FastAPI backend URL
BACKEND_URL = "http://localhost:8001/chat"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("AI Assistant Chat")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send the message to the backend
    session_id = st.session_state.session_id
    response = requests.post(
        BACKEND_URL,
        json={"session_id": session_id, "message": prompt}
    )
    response_json = response.json()
    ai_message = response_json["message"]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(ai_message)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_message})
