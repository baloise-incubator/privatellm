"""Program to play with langchain"""

import logging
import os
import tempfile
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urljoin

import httpx
import requests
import uvicorn
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain.agents import create_sql_agent, initialize_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import get_openai_callback
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
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

if TYPE_CHECKING:
    from langchain.document_loaders.unstructured import UnstructuredBaseLoader
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.vectorstores.pgvector import PGVector
from miniopy_async import Minio
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from sqlalchemy import create_engine, text

# Map file extensions to document loaders and their arguments
LOADER_MAPPING: dict[str, tuple[Any, dict[str, str]]] = {
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

load_dotenv()  # take environment variables from .env

# Postgres connection string
CONNECTION_STRING = os.getenv("PGVECTOR_DB", "")
# S3 settinga
S3_BUCKET = os.getenv("S3_BUCKET", "texttitan")
S3_URL = os.getenv("S3_URL", "localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")

s3client = Minio("localhost:9000", access_key=S3_ACCESS_KEY, secret_key=S3_SECRET_KEY, secure=False)


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


app = FastAPI(title="TextTitan")


# Define basic authentication security object
security = HTTPBasic()

# Users (for demonstration purposes, replace with your own authentication logic)
fake_users_db = {"rrr": "rrr", "ttt": "ttt", "yyy": "yyy"}


# opentelemetry instrumentation
FastAPIInstrumentor.instrument_app(app)
Psycopg2Instrumentor().instrument(enable_commenter=True, commenter_options={})
LoggingInstrumentor().instrument(set_logging_format=True)
HTTPXClientInstrumentor().instrument()
AioHttpClientInstrumentor().instrument()
RequestsInstrumentor().instrument()
resource = Resource(attributes={"service.name": "texttitan"})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)
endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
if endpoint:
    span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    provider.add_span_processor(span_processor)


# Verify user credentials
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    user = fake_users_db.get(credentials.username)
    if user is None or user != credentials.password:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# Custom authentication dependency decorator
def authenticate_user(username: str = Depends(verify_credentials)) -> str:
    return username


def load_single_document(file_path: Path) -> list[Document]:
    ext = file_path.suffix.lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader: UnstructuredBaseLoader = loader_class(str(file_path), **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


async def update_files(
    filenames: list[Path], documenttype: DocumentTypeEnum, username: str, source: str | None = None
) -> None:
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

                document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
                documents = list(document_transformer.transform_documents(documents))

        await update_embedding(documents, username)


async def update_embedding(documents: list[Document], username: str) -> None:
    embeddings = OpenAIEmbeddings()
    db = PGVector(
        embedding_function=embeddings,
        collection_name=username,
        connection_string=CONNECTION_STRING,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    await db.aadd_documents(docs)


@app.get("/files/{userpath}")
async def get_files(userpath: str, username: str = Depends(authenticate_user)) -> list[dict[str, Any]]:
    # Get the file path and serve it
    if userpath != username:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden to access /files/{userpath}",
        )
    prefix = Path("files") / userpath
    objects = await s3client.list_objects(S3_BUCKET, start_after=str(prefix), recursive=True)
    return [
        {
            "object_name": obj.object_name,
            "bucket_name": obj.bucket_name,
            "content_type": obj.content_type,
            "etag": obj.etag,
            "last_modified": obj.last_modified,
            "size": obj.size,
        }
        for obj in objects
    ]


@app.get("/files/{userpath}/{filename}")
async def get_file(userpath: str, filename: str, username: str = Depends(authenticate_user)) -> RedirectResponse:
    # Get the file path and serve it
    if userpath != username:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden to access /files/{userpath}/{filename}",
        )
    file_path = Path("files") / userpath / filename
    url = await s3client.get_presigned_url(
        "GET",
        S3_BUCKET,
        str(file_path),
        expires=timedelta(hours=1),
    )

    if url is None:
        raise HTTPException(
            status_code=404,
            detail=f"No such file /files/{userpath}/{filename}",
        )
    return RedirectResponse(url)


@app.delete("/files/{userpath}/{filename}")
async def delete_file(userpath: str, filename: str, username: str = Depends(authenticate_user)) -> str:
    # Get the file path and serve it
    if userpath != username:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden to access /files/{userpath}/{filename}",
        )
    file_path = Path("files") / userpath / filename
    objects = await s3client.list_objects(S3_BUCKET, prefix=str(file_path))
    if not objects:
        raise HTTPException(
            status_code=404,
            detail=f"No such file /{file_path}",
        )
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as session:
        session.execute(
            text(
                """delete from langchain_pg_embedding
                where cmetadata ->> 'source' = :path
                and collection_id  =
                (select uuid from langchain_pg_collection where name = :username);"""
            ).bindparams(path=str(file_path), username=username)
        )
        session.commit()
    await s3client.remove_object(S3_BUCKET, str(file_path))
    return ""


@app.post("/files/")
async def upload_files(
    pdf_files: list[UploadFile],
    documenttype: DocumentTypeEnum,
    username: str = Depends(authenticate_user),
) -> dict[str, list[Path]]:
    filenames = []
    dirname = Path("files") / username
    for pdf_file in pdf_files:
        fn = dirname / cast(str, pdf_file.filename)
        await s3client.put_object(bucket_name=S3_BUCKET, object_name=str(fn), data=pdf_file.file, length=pdf_file.size)
        filenames.append(fn)
    await update_files(filenames, documenttype, username)
    return {"uploaded_filenames": filenames}


@app.post("/websites/")
async def ingest_website(url: str, username: str = Depends(authenticate_user)) -> None:
    await scrape_website(url, username)


async def scrape_website(url: str, username: str, depth: int = 1, visited: set[str] | None = None) -> None:
    if visited is None:
        visited = set()

    if url in visited or depth < 0 or len(visited) > 1000:  # noqa: PLR2004
        return

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=3)
            response.raise_for_status()

        # Save content to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, mode="w", encoding="utf-8", suffix=".html") as tmp:
            tmp.write(response.text)
            tmp.flush()
            await update_files([Path(tmp.name)], DocumentTypeEnum.RANDOM, username, url)
        visited.add(url)

        # If we haven't reached our desired depth, continue recursively
        if depth > 0:
            links = get_links_from_url(url)
            for link in links:
                absolute_link = urljoin(url, link)
                await scrape_website(absolute_link, username, depth - 1, visited)
    except requests.RequestException:
        logging.exception("Error fetching URL %s", url)


def get_links_from_url(url: str) -> list[str]:
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        return [a["href"] for a in soup.find_all("a", href=True)]
    except requests.RequestException:
        logging.exception("Error fetching URL %s", url)
        return []


@app.post("/chat/")
async def chat(query: str, model_enum: ModelEnum = ModelEnum.LLAMA) -> Any:
    """
    Generates a response by interacting with a language model using the query.

    Args:
        query (str): The user's query or question.
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
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",  # type: ignore[call-arg]
                temperature=0.75,
                max_tokens=2000,
                callback_manager=callback_manager,
                verbose=True,  # Verbose is required to pass to the callback manager
            )
        case _:
            raise ValueError(f"unsupported model {model_enum}")
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    tracer = trace.get_tracer(__name__)
    with get_openai_callback() as cb, tracer.start_as_current_span("chat") as span:
        resp = await llm_chain.arun(query)
        span.set_attributes(
            {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost_usd": cb.total_cost,
            }
        )
        return resp.strip()


@app.post("/chat_with_chinook/")
async def chat_with_chinook(
    question: str,
) -> Any:
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
    # downloaded from https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip
    db = SQLDatabase.from_uri("sqlite:///db/chinook.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))
    agent_executor = create_sql_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    tracer = trace.get_tracer(__name__)
    with get_openai_callback() as cb, tracer.start_as_current_span("chat_with_chinook") as span:
        resp = await agent_executor.arun(question)
        span.set_attributes(
            {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost_usd": cb.total_cost,
            }
        )
        return resp.strip()


async def load_db(username: str) -> PGVector:
    """
    Loads and populates a database for the given username.

    Args:
        username (str): The username for database identification.

    Returns:
        PGVector: The populated database instance.

    Example:
    >>> loaded_db = await load_db("john_doe")
    """
    embeddings = OpenAIEmbeddings()
    return PGVector(
        embedding_function=embeddings,
        collection_name=username,
        connection_string=CONNECTION_STRING,
    )


async def query_db(question: str, username: str) -> list[Document]:
    db = await load_db(username)
    return db.similarity_search(question)


async def query_db_with_type(question: str, username: str, doctype: DocumentTypeEnum) -> list[Document]:
    db = await load_db(username)
    return db.similarity_search(question, filter={"documenttype": doctype.name})


@app.post("/chat_with_documents/")
async def chat_with_documents(query: str, request: Request, username: str = Depends(authenticate_user)) -> Any:
    """
    Generates a response by interacting with a language model using query and user documents.

    Args:
        query (str): The user's query or question.
    """
    template = """Please give a short answer using the context enclosed in <ctx></ctx>.
    If the context does not contain the information respond with "texttitan cannot help you with that".

    <ctx>
    {summaries}
    </ctx>

    question: {question}. Always give the source document with the complete file path

    answer:"""

    docs = await query_db(query, username)
    prompt = PromptTemplate(template=template, input_variables=["question", "summaries"])
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # type: ignore[call-arg]
        temperature=0.75,
        max_tokens=500,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    tracer = trace.get_tracer(__name__)
    with get_openai_callback() as cb, tracer.start_as_current_span("chat_with_documents") as span:
        resp = await llm_chain.arun({"question": query, "summaries": docs})
        span.set_attributes(
            {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost_usd": cb.total_cost,
            }
        )
        return resp.replace(f"files/{username}/", f"{request.base_url}files/{username}/")


@app.post("/chat_with_agent/")
async def chat_with_agent(query: str, request: Request, username: str = Depends(authenticate_user)) -> Any:
    """
    API endpoint for chatting with an agent.

    This endpoint allows a user to chat with a model-agent which can perform various tasks like retrieving bills,
    random documents, and reminders based on the query provided by the user. The response from the agent
    will be an observation including its source, without any interpretation.

    Args:
        query (str): The query or prompt provided by the user.
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
    """  # noqa: E501
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

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # type: ignore[call-arg]
        temperature=0.75,
        max_tokens=500,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    respond_tool = Tool.from_function(
        func=lambda x: parsing_llm(x, query),
        name="Responder",
        description="""This tool sould always be called as the last tool.
The documents have to be retrieved using another tool beforehand. It requires the documents as its input""",
        coroutine=lambda x: parsing_llm(x, query),
    )
    tools = [get_bills_tool, get_randoms_tool, get_reminders_tool, respond_tool]

    agent = initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        verbose=True,
    )

    tracer = trace.get_tracer(__name__)
    with get_openai_callback() as cb, tracer.start_as_current_span("chat_with_agent") as span:
        resp = await agent.arun(
            {"query": query + " Just return the observation including its source without interpreting it."}
        )
        span.set_attributes(
            {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost_usd": cb.total_cost,
            }
        )
        return resp.replace(f"files/{username}/", f"{request.base_url}files/{username}/")


async def get_bills(query: str, username: str) -> list[Document]:
    return await query_db_with_type(query, username, DocumentTypeEnum.BILL)


async def get_randoms(query: str, username: str) -> list[Document]:
    return await query_db_with_type(query, username, DocumentTypeEnum.RANDOM)


async def get_reminders(query: str, username: str) -> list[Document]:
    return await query_db_with_type(query, username, DocumentTypeEnum.REMINDER)


async def parsing_llm(query: str, question: str) -> Any:
    template = """Please give a short answer using the context enclosed in <ctx></ctx>.
    If the context does not contain the information respond with "texttitan cannot help you with that".

    <ctx>
    {summaries}
    </ctx>

    question: {question}. Always give the source document with the complete file path

    answer:"""
    prompt = PromptTemplate(template=template, input_variables=["question", "summaries"])
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # type: ignore[call-arg]
        temperature=0.75,
        max_tokens=2000,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return await llm_chain.arun({"question": question, "summaries": query})


async def main() -> None:
    bucket_exists = await s3client.bucket_exists(S3_BUCKET)
    if not bucket_exists:
        await s3client.make_bucket(S3_BUCKET)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)  # noqa: S104


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
