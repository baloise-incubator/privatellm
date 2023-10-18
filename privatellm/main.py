"""Program to play with langchain"""

import os
import glob
import logging
from typing import List
from enum import Enum
from timeit import default_timer as timer
import uvicorn
from fastapi import FastAPI, UploadFile, Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import LlamaCpp, GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


# Configure the logging module
logging.basicConfig(
    level=logging.DEBUG,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ModelEnum(Enum):
    LLAMA = "llama"
    GPT4ALL = "gpt4all"
    CHATGPT = "chatgpt"


app = FastAPI()

# Define basic authentication security object
security = HTTPBasic()

# Users (for demonstration purposes, replace with your own authentication logic)
fake_users_db = {"rrr": "rrr", "ttt": "ttt", "yyy": "yyy"}


def assert_api_key():
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is not None:
        return
    if os.path.exists("apikey.txt"):
        api_key = open("apikey.txt", "r", encoding="utf-8").read().strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    return api_key


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


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

async def update_files(filenames, username):
    assert_api_key()
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="./db", embedding_function=embeddings, collection_name=username)
    for fn in filenames:
        logging.info("generate embeddings for %s", fn)
        documents = load_single_document(fn)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)

        collection = db.get(where={"source": fn})
        if collection:
            # await db.update_document()
            await db.aadd_documents(docs)
        else:
            await db.aadd_documents(docs)
        logging.info("persisting")
    return db

@app.post("/file/")
async def upload_files(
    pdf_files: List[UploadFile], username: str = Depends(authenticate_user)
):
    filenames = []
    dirname = f"uploads/{username}"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    for pdf_file in pdf_files:
        fn = f"{dirname}/{pdf_file.filename}"
        with open(fn, "wb") as f:
            f.write(pdf_file.file.read())
        filenames.append(fn)
    await update_files(filenames, username)
    return {"uploaded_filenames": filenames}


@app.post("/chat/")
async def chat(
    input: str,
    model_enum: ModelEnum = ModelEnum.LLAMA,
    username: str = Depends(authenticate_user),
):
    template = """Question: {question}

    Answer: Give a short answer."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    match model_enum:
        case ModelEnum.LLAMA:
            llm = LlamaCpp(
                # model downloaded from meta and
                # converted with https://github.com/ggerganov/llama.cpp/blob/master/convert.py
                model_path="llama-2-7b-chat/ggml-model-f16.gguf",
                temperature=0.75,
                max_tokens=2000,
                top_p=1,
                callback_manager=callback_manager,
                verbose=True,  # Verbose is required to pass to the callback manager
            )
        case ModelEnum.GPT4ALL:
            template = """"Question: {question}

                Answer: Let's think step by step."""

            prompt = PromptTemplate(template=template, input_variables=["question"])
            llm = GPT4All(
                model="gpt4all/model.bin",
                backend="gptj",
                callback_manager=callback_manager,
                verbose=True,
            )
        case ModelEnum.CHATGPT:
            api_key = assert_api_key()

            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.75,
                max_tokens=2000,
                top_p=1,
                callback_manager=callback_manager,
                verbose=True,  # Verbose is required to pass to the callback manager
            )
        case _:
            raise ValueError(f"unsupported model {model_enum}")
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    resp = await llm_chain.arun(input)
    return resp.strip()


async def populate_db(db, username):
    collection = db.get()
    existing_docs = set([metadata["source"] for metadata in collection["metadatas"]])
    logging.info("existing documents %s", existing_docs)
    for fn in glob.glob(f"data/{username}/*.txt"):
        if fn in existing_docs:
            continue
        loader = TextLoader(fn)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        logging.info("generate embeddings for %s", fn)
        await db.aadd_documents(docs)
        logging.info("persisting")
        db.persist()


async def load_db(username: str):
    start = timer()
    assert_api_key()
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="./db", embedding_function=embeddings, collection_name=username)
    await populate_db(db, username)
    print(f'ingestion took {timer() - start}')
    return db


async def query_db(question: str, username: str):
    db = await load_db(username)
    docs = db.similarity_search(question)
    return docs


@app.post("/chat_with_documents/")

async def chat_with_documents(input: str, username: str = Depends(authenticate_user)):
    template = """Please give a short answer using the context enclosed in <ctx></ctx> adding the source of the document used to respond.
    If the context does not contain the information respond with "texttitan does not want to help".

    <ctx>
    {summaries}
    </ctx>

    question: {question}

    answer:"""

    docs = await query_db(input, username)
    prompt = PromptTemplate(
        template=template, input_variables=["question", "summaries"]
    )
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    api_key = assert_api_key()

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo-16k",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    start = timer()
    resp = await llm_chain.arun({"question": input, "summaries": docs})
    print(f'inference took {timer() - start}')
    return resp.replace(f"uploads/{username}/", f"{request.base_url}/{username}/")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)
