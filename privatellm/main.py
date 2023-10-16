from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import List
import uvicorn
from langchain.agents.agent_toolkits.openapi import planner
import os
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = FastAPI()

# Define basic authentication security object
security = HTTPBasic()

# Users (for demonstration purposes, replace with your own authentication logic)
fake_users_db = {
    "rr": "rr"
}

def assert_api_key():
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is not None:
        return
    if os.path.exists("apikey.txt"):
        api_key = open("apikey.txt", "r").read().strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Verify user credentials
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    user = fake_users_db.get(credentials.username)
    if user is None or user != credentials.password:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Custom authentication dependency decorator
def authenticate_user(username: str = Depends(verify_credentials)):
    return username

@app.post("/uploadpdf/")
async def upload_pdfs(
    pdf_files: List[UploadFile],
    username: str = Depends(authenticate_user)
):
    filenames = []
    for pdf_file in pdf_files:
        with open(f"uploads/{pdf_file.filename}", "wb") as f:
            f.write(pdf_file.file.read())
        filenames.append(pdf_file.filename)

    return {"uploaded_filenames": filenames}

@app.post("/chat/")
async def chat_trial(
    input: str,
    username: str = Depends(authenticate_user)
):
    template = """Question: {question}

    Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager, 
        verbose=True, # Verbose is required to pass to the callback manager
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)