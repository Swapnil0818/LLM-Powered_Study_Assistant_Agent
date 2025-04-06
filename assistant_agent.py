from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
import chainlit as cl

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

api_key='<API_KEY>'
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
vector_store = Chroma(embedding_function = embeddings)

@cl.on_chat_start
async def on_chat_start():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key,streaming = True)
    cl.user_session.set("chat_history", [])
    
    loader = TextLoader("RAG_Documents\\A_Study_In_Scarlet.txt", encoding="utf-8")

    documents = loader.load()
    print(documents[0])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # text_splitter1 = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)
    
    ## Load document chunks to vectorDB
    _ = vector_store.add_documents(documents=all_splits)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are to answer the user query based on the context provided as follows:\n\n {context}",
            ),
            ("human", "{chat_history}\nHuman: {question}"),
        ]
    )
    
    
    
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    
    chat_history = cl.user_session.get("chat_history", [])
    
    # If it's the first message, store it immediately in history
    if not chat_history:
        chat_history.append(f"Human: {message.content}")
    
    msg = cl.Message(content="")
    
    results = vector_store.similarity_search(message.content)
    i = 1
    context = ""
    for res in results:
        context += "Context Document " + str(i) + ":\n" + res.page_content + "\n\n"
        i += 1

    async for chunk in runnable.astream(
        {"context": context, "chat_history": "\n".join(chat_history), "question": message.content},  # Ensure chat_history is passed
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    chat_history.append(f"Pirate AI: {msg.content}")
    cl.user_session.set("chat_history", chat_history)
    await msg.send()