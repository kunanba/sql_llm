from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import your existing code
from my_ai_agent import app as langgraph_app  # Ensure this imports the initialized app

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add logging statements
logger.info("Backend server is running.")


# Initialize FastAPI app
app = FastAPI()

# Define request and response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    user_message = chat_request.message

    # Create the initial state with the user's message
    state = {"messages": [("human", user_message)]}

    # Invoke the LangGraph app with the state
    output = langgraph_app.invoke(state)

    # Extract the AI's response from the output
    ai_message_content = output["messages"][-1].content
    logger.info(f"Received message: {user_message}")
    # Return the AI's message
    return ChatResponse(message=ai_message_content)
