"""Program to play with langchain"""

import os
import glob
import logging
from typing import List, Dict, Tuple, Any, Optional
from enum import Enum
from timeit import default_timer as timer
from urllib.parse import urljoin
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain.chat_models import ChatOpenAI
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
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
from langchain.agents import initialize_agent
from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup


os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Map file extensions to document loaders and their arguments
LOADER_MAPPING: Dict[str, Tuple[Any, Dict[str, str]]] = {
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
    """Enum of available models."""
    LLAMA = "llama"
    GPT4ALL = "gpt4all"
    CHATGPT = "chatgpt"


class DocumentTypeEnum(Enum):
    """Document type"""

    BILL = "bill"
    REMINDER = "reminder"
    RANDOM = "random"


app = FastAPI()

# Define basic authentication security object
security = HTTPBasic()

# Users (for demonstration purposes, replace with your own authentication logic)
fake_users_db = {"rrr": "rrr", "ttt": "ttt", "yyy": "yyy"}


def assert_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is not None:
        return api_key
    if os.path.exists("apikey.txt"):
        with open("apikey.txt", "r", encoding="utf-8") as f:
            api_key = f.read().strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return api_key
    raise AssertionError('Missing OpenAI API key.')


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


async def update_files(filenames, documenttype, username, source: Optional[str] = None):
    for fn in filenames:
        logging.info("generate embeddings for %s", fn)
        documents = load_single_document(fn)
        if source:
            # overwrite source with actual source
            for doc in documents:
                doc.metadata["source"] = source
        if documenttype:
            for doc in documents:
                doc.metadata["documenttype"] = documenttype.name
            if documenttype == DocumentTypeEnum.BILL:
                assert_api_key()
                schema = {
                    "properties": {
                        "name": {"type": "string"},
                        "Vertragskonto-Nr": {"type": "string"},
                        "Rechnungsbetrag": {"type": "string"},
                        "Zahlungsart:": {
                            "type": "string",
                            "enum": ["eBill", "Rechnung"],
                        },
                    },
                    "required": ["name", "Rechnungsbetrag", "Zahlungsart"],
                }

                # Must be an OpenAI model that supports functions
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

                document_transformer = create_metadata_tagger(
                    metadata_schema=schema, llm=llm
                )
                documents = list(document_transformer.transform_documents(documents))

        await update_embedding(documents, username)


async def update_embedding(documents: List[Document], username: str) -> None:
    assert_api_key()
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        persist_directory="./db",
        embedding_function=embeddings,
        collection_name=username,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    await db.aadd_documents(docs)


@app.get("/files/{userpath}")
async def get_files(userpath: str, username: str = Depends(authenticate_user)):
    # Get the file path and serve it
    if userpath != username:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden to access /files/{userpath}",
        )
    files = glob.glob(f"files/{userpath}/*")
    return str(files)


@app.get("/files/{userpath}/{filename}")
async def get_file(
    userpath: str, filename: str, username: str = Depends(authenticate_user)
):
    # Get the file path and serve it
    if userpath != username:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden to access /files/{userpath}/{filename}",
        )
    file_path = os.path.join("files", userpath, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"No such file /files/{userpath}/{filename}",
        )
    return FileResponse(file_path)


@app.delete("/files/{userpath}/{filename}")
async def delete_file(
    userpath: str, filename: str, username: str = Depends(authenticate_user)
):
    # Get the file path and serve it
    if userpath != username:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden to access /files/{userpath}/{filename}",
        )
    file_path = os.path.join("files", userpath, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"No such file /files/{userpath}/{filename}",
        )
    db = Chroma(persist_directory="./db", collection_name=username)
    collection = db.get(where={"source": f"files/{userpath}/{filename}"})
    if collection["ids"]:
        db._collection.delete(ids=collection["ids"])  # pylint: disable=protected-access
    os.remove(os.path.join("files", userpath, filename))
    return ""


@app.post("/files/")
async def upload_files(
    pdf_files: List[UploadFile],
    documenttype: DocumentTypeEnum,
    username: str = Depends(authenticate_user),
):
    filenames = []
    dirname = f"files/{username}"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    for pdf_file in pdf_files:
        fn = f"{dirname}/{pdf_file.filename}"
        with open(fn, "wb") as f:
            f.write(pdf_file.file.read())
        filenames.append(fn)
    await update_files(filenames, documenttype, username)
    return {"uploaded_filenames": filenames}


@app.post("/websites/")
async def ingest_website(url: str, username: str = Depends(authenticate_user)):
    await scrape_website(url, username)


async def scrape_website(url, username, depth=1, visited=None):
    if visited is None:
        visited = set()

    if url in visited or depth < 0 or len(visited) > 1000:
        return

    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()

        # Save content to a temporary file
        with tempfile.NamedTemporaryFile(
            delete=True, mode="w", encoding="utf-8", suffix=".html"
        ) as tmp:
            tmp.write(response.text)
            await update_files([tmp.name], DocumentTypeEnum.RANDOM, username, url)
        visited.add(url)

        # If we haven't reached our desired depth, continue recursively
        if depth > 0:
            links = get_links_from_url(url)
            for link in links:
                absolute_link = urljoin(url, link)
                await scrape_website(absolute_link, username, depth - 1, visited)
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")


def get_links_from_url(url):
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True)]
        return links
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return []


@app.post("/chat/")
async def chat(
    input: str,
    model_enum: ModelEnum = ModelEnum.LLAMA,
    username: str = Depends(authenticate_user),
):
    """
    Generates a response by interacting with a language model using the input.

    Args:
        input (str): The user's input or question.
    """
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
                model_name="gpt-3.5-turbo",  # type: ignore[call-arg]
                temperature=0.75,
                max_tokens=2000,
                top_p=1,  # type: ignore[call-arg]
                callback_manager=callback_manager,
                verbose=True,  # Verbose is required to pass to the callback manager
            )
        case _:
            raise ValueError(f"unsupported model {model_enum}")
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    resp = await llm_chain.arun(input)
    return resp.strip()


@app.post("/chat_with_chinook/")
async def chat_with_chinook(
    question: str,
):
    """
    This function processes user questions using a combination of a language model
    and an SQLite database named 'chinook.db'. It generates responses based on the
    user's input and generated database queries.

    Parameters:
        question (str): The user's input question.

    Returns:
        str: A response generated by the system based on the user's question.

    Examples:
        - how many customers as there
        - give me the most important kpi values for each customer
        - give me the lexically first 3 names by country using a window function for the country
    """
    assert_api_key()

    # downloaded from https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip
    db = SQLDatabase.from_uri("sqlite:///db/chinook.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))
    agent_executor = create_sql_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    resp = await agent_executor.arun(question)
    return resp.strip()


async def populate_db(db, username):
    """
    Populates a database with documents for a given username.

    Args:
        db (Database): The target database.
        username (str): The username indicating the data directory.

    This function populates the database with documents from the 'data/{username}' directory.
    If a document with the same filename already exists, it is skipped.

    Example:
    >>> db = Database()
    >>> await populate_db(db, 'john_doe')
    """
    collection = db.get()
    existing_docs = {metadata["source"] for metadata in collection["metadatas"]}
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
    """
    Loads and populates a database for the given username.

    Args:
        username (str): The username for database identification.

    Returns:
        Chroma: The populated database instance.

    Example:
    >>> loaded_db = await load_db("john_doe")
    """
    start = timer()
    assert_api_key()
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        persist_directory="./db",
        embedding_function=embeddings,
        collection_name=username,
    )
    await populate_db(db, username)
    print(f"ingestion took {timer() - start}")
    return db


async def query_db(question: str, username: str):
    db = await load_db(username)
    docs = db.similarity_search(question)
    return docs


async def query_db_with_type(question: str, username: str, type: DocumentTypeEnum):
    db = await load_db(username)
    docs = db.similarity_search(question, filter={"documenttype": type.name})
    return docs


@app.post("/chat_with_documents/")
async def chat_with_documents(
    input: str, request: Request, username: str = Depends(authenticate_user)
):
    """
    Generates a response by interacting with a language model using input and user documents.

    Args:
        input (str): The user's input or question.
    """
    template = """Please give a short answer using the context enclosed in <ctx></ctx>.
    If the context does not contain the information respond with "texttitan cannot help you with that".

    <ctx>
    {summaries}
    </ctx>

    question: {question}. Always give the source document with the complete file path

    answer:"""

    docs = await query_db(input, username)
    prompt = PromptTemplate(
        template=template, input_variables=["question", "summaries"]
    )
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    api_key = assert_api_key()

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",  # type: ignore[call-arg]
        temperature=0.75,
        max_tokens=500,
        top_p=1,  # type: ignore[call-arg]
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    start = timer()
    resp = await llm_chain.arun({"question": input, "summaries": docs})
    print(f"inference took {timer() - start}")
    return resp.replace(f"files/{username}/", f"{request.base_url}files/{username}/")


@app.post("/chat_with_agent/")
async def chat_with_agent(
    input: str, request: Request, username: str = Depends(authenticate_user)
):
    """
    API endpoint for chatting with an agent.

    This endpoint allows a user to chat with a model-agent which can perform various tasks like retrieving bills,
    random documents, and reminders based on the input query provided by the user. The response from the agent
    will be an observation including its source, without any interpretation.

    Args:
        input (str): The input question or prompt provided by the user.
        request (Request): The incoming request object.
        username (str, optional): The username of the authenticated user. It is ensured by the `authenticate_user` dependency.

    Returns:
        str: A response from the agent, where the file paths in the response are replaced with the full URL paths.

    Usage:
        To use this endpoint, make a POST request to `/chat_with_agent/` with the required parameters.

    Notes:
        - It uses three tools, `GetBills`, `GetRandom`, and `GetReminders` to fetch relevant documents based on the input query.
        - The `Responder` tool is used to parse and format the agent's response before sending it back to the user.
        - This endpoint leverages the ChatOpenAI model (specifically "gpt-3.5-turbo") for inference.
        - The response time for the endpoint (inference time) will be printed.
    """
    get_bills_tool = Tool.from_function(
        func=lambda x: get_bills(x, username),
        name="GetBills",
        description="Returns documents of type bill. The input should be the question.",
        coroutine=lambda x: get_bills(x, username),
    )
    get_randoms_tool = Tool.from_function(
        func=lambda x: get_randoms(x, username),
        name="GetRandom",
        description="Returns documents of type random. The input should be the question.",
        coroutine=lambda x: get_randoms(x, username),
    )
    get_reminders_tool = Tool.from_function(
        func=lambda x: get_reminders(x, username),
        name="GetReminders",
        description="Returns documents of type reminder. The input should be the question.",
        coroutine=lambda x: get_reminders(x, username),
    )

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    api_key = assert_api_key()

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",  # type: ignore[call-arg]
        temperature=0.75,
        max_tokens=500,
        top_p=1,  # type: ignore[call-arg]
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    respond_tool = Tool.from_function(
        func=lambda x: parsing_llm(x, input),
        name="Responder",
        description="This tool sould always be called as the last tool. The documents have to be retrieved using another tool beforehand. It requires the documents as its input",
        coroutine=lambda x: parsing_llm(x, input),
    )
    tools = [get_bills_tool, get_randoms_tool, get_reminders_tool, respond_tool]

    agent = initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        verbose=True,
    )

    start = timer()
    resp = await agent.arun(
        {
            "input": input
            + " Just return the observation including its source without interpreting it."
        }
    )
    print(f"inference took {timer() - start}")
    return resp.replace(f"files/{username}/", f"{request.base_url}files/{username}/")


async def get_bills(input: str, username: str):
    return await query_db_with_type(input, username, DocumentTypeEnum.BILL)


async def get_randoms(input: str, username: str):
    return await query_db_with_type(input, username, DocumentTypeEnum.RANDOM)


async def get_reminders(input: str, username: str):
    return await query_db_with_type(input, username, DocumentTypeEnum.REMINDER)


async def parsing_llm(input: str, question: str):
    template = """Please give a short answer using the context enclosed in <ctx></ctx>.
    If the context does not contain the information respond with "texttitan cannot help you with that".

    <ctx>
    {summaries}
    </ctx>

    question: {question}. Always give the source document with the complete file path

    answer:"""
    prompt = PromptTemplate(
        template=template, input_variables=["question", "summaries"]
    )
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    api_key = assert_api_key()

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",  # type: ignore[call-arg]
        temperature=0.75,
        max_tokens=2000,
        top_p=1,  # type: ignore[call-arg]
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    resp = await llm_chain.arun({"question": question, "summaries": input})
    return resp


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)
