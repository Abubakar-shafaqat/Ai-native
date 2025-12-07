from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg
from contextlib import asynccontextmanager
import google.generativeai as genai
from qdrant_client import models
import logging
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Dict, Optional, Any
from fastapi import Depends, HTTPException
from datetime import timedelta

from .auth import (
    get_password_hash, verify_password, create_access_token, decode_access_token, users_db, UserInDB,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)

from config import DATABASE_URL, GEMINI_API_KEY
from qdrant_client_lib import initialize_qdrant_collection, close_qdrant_client, get_qdrant_client, QDRANT_COLLECTION_NAME
from ingest import ingest_book_content, generate_embedding

# --- Global Variables ---
db_pool = None
gemini_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Event ---
    logging.info("Application startup initiated.")
    global db_pool, gemini_model

    # Log environment variables
    logging.info(f"DATABASE_URL: {DATABASE_URL}")
    logging.info(f"GEMINI_API_KEY is set: {bool(GEMINI_API_KEY)}")


    # Initialize PostgreSQL
    logging.info("Connecting to PostgreSQL database...")
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        logging.info("PostgreSQL connection pool created.")
    except Exception as e:
        logging.error(f"Failed to create PostgreSQL connection pool: {e}")
        db_pool = None

    # Initialize Qdrant
    logging.info("Connecting to Qdrant...")
    try:
        await initialize_qdrant_collection()
        logging.info("Qdrant initialized and collection checked.")
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant: {e}")

    # Initialize Gemini Client
    logging.info("Initializing Gemini client...")
    if not GEMINI_API_KEY:
        logging.critical("CRITICAL: GEMINI_API_KEY is not set. Please create a .env file with your API key.")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logging.info("Gemini client initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini client: {e}")
            gemini_model = None

    yield

    # --- Shutdown Event ---
    logging.info("Application shutdown initiated.")
    if db_pool:
        logging.info("Closing PostgreSQL connection pool.")
        await db_pool.close()
        logging.info("PostgreSQL connection pool closed.")
    
    close_qdrant_client()
    logging.info("Qdrant client closed.")
    logging.info("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)





# Pydantic models for authentication


class UserCreate(BaseModel):


    username: str


    email: str


    password: str





class UserLogin(BaseModel):


    username: str


    password: str





class Token(BaseModel):


    access_token: str


    token_type: str





class User(BaseModel):


    username: str


    email: str


    full_name: Optional[str] = None





oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")





@app.post("/auth/signup", response_model=User)


async def signup(user_data: UserCreate):


    if user_data.username in users_db:


        raise HTTPException(status_code=400, detail="Username already registered")


    


    hashed_password = get_password_hash(user_data.password)


    users_db[user_data.username] = {


        "hashed_password": hashed_password,


        "email": user_data.email,


        "full_name": user_data.username # Using username as full_name for simplicity


    }


    logging.info(f"User {user_data.username} registered.")


    return User(username=user_data.username, email=user_data.email, full_name=user_data.username)





@app.post("/auth/login", response_model=Token)


async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):


    user = users_db.get(form_data.username)


    if not user or not verify_password(form_data.password, user["hashed_password"]):


        raise HTTPException(


            status_code=401,


            detail="Incorrect username or password",


            headers={"WWW-Authenticate": "Bearer"},


        )


    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)


    access_token = create_access_token(


        data={"sub": form_data.username}, expires_delta=access_token_expires


    )


    logging.info(f"User {form_data.username} logged in.")


    return {"access_token": access_token, "token_type": "bearer"}





@app.get("/auth/me", response_model=User)


async def read_users_me(token: str = Depends(oauth2_scheme)):


    payload = decode_access_token(token)


    if payload is None:


        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


    


    username: str = payload.get("sub")


    if username is None or username not in users_db:


        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


    


    user_data = users_db[username]


    return User(username=username, email=user_data["email"], full_name=user_data["full_name"])





# Pydantic model for chat request
class ChatRequest(BaseModel):
    question: str
    selected_text: str = None # Optional selected text for contextual queries

@app.get("/")
async def read_root():
    return {"message": "Chatbot Backend is running!"}

@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini client not initialized.")
    if not get_qdrant_client():
        raise HTTPException(status_code=500, detail="Qdrant client not initialized.")

    user_query = request.question
    context_text = request.selected_text

    # 1. Generate embedding for the user query (now using Gemini)
    query_embedding = await generate_embedding(user_query)
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

    # 2. Retrieve relevant documents from Qdrant
    try:
        qdrant_results = await get_qdrant_client().search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=5,
            with_payload=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant search failed: {e}")

    # 3. Assemble context for the LLM
    retrieved_chunks = [result.payload["text"] for result in qdrant_results if result.payload and "text" in result.payload]
    
    if context_text:
        combined_context = f"User selected text for context:\n{context_text}\n\nRelevant document chunks:\n" + "\n---".join(retrieved_chunks)

    else:
        combined_context = "Relevant document chunks:\n" + "\n---".join(retrieved_chunks)

    # 4. Generate response using LLM (Gemini)
    try:
        prompt = (
            "You are a helpful assistant specialized in Physical AI & Humanoid Robotics. "
            "Answer the user's question based ONLY on the provided context. "
            "If the answer cannot be found in the context, state that you don't have enough information.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question: {user_query}"
        )
        
        response = await gemini_model.generate_content_async(prompt)
        
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")