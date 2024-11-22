# app.py

import streamlit as st
import requests
import uuid
import base64
from io import BytesIO

# Set the FastAPI backend URL
BACKEND_URL = "http://localhost:8000/chat"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("AI Assistant Chat")

# Display chat messages from history on app rerun
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display images if any
        for image_base64 in message.get("images", []):
            image_bytes = base64.b64decode(image_base64)
            st.image(image_bytes)

# Accept user input
if prompt := st.chat_input("Ask a question"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    # Send the message to the backend
    session_id = st.session_state.session_id
    response = requests.post(
        BACKEND_URL,
        json={"session_id": session_id, "message": prompt}
    )
    response_json = response.json()
    ai_message = response_json["message"]
    images = response_json.get("images", [])

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(ai_message)
        # Display images if any
        for image_base64 in images:
            image_bytes = base64.b64decode(image_base64)
            st.image(image_bytes)
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_message,
        "images": images
    })
