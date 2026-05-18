import argparse
from collections.abc import Iterable
from typing import Any

try:
    import readline  # noqa: F401
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
CHROMA_PATH = "chroma-local"
DOCS_PATH = "swift-book-documentation"
DOCS_GLOB = "**/*.md"
EMBEDDING_MODEL = "embeddinggemma"
CHAT_MODEL = "qwen2.5-coder:3b"
RESULT_COUNT = 2

SYSTEM_PROMPT = """You are AskSwift, a helpful Swift documentation assistant.
Use the Swift documentation excerpts to ground your answers.
If the excerpts do not contain enough information, say so instead of guessing."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def get_vector_store() -> Chroma:
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    if db._collection.count() > 0:
        return db

    print("Building local Swift documentation index. This may take a few minutes.")
    loader = DirectoryLoader(
        DOCS_PATH,
        glob=DOCS_GLOB,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    db.add_documents(chunks)
    return db


def response_text(response: Any) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def format_response(response: Any, sources: Iterable[str | None]) -> str:
    lines = [line.rstrip() for line in response_text(response).strip().splitlines()]
    clean_lines = []
    blank_count = 0

    for line in lines:
        if line:
            clean_lines.append(line)
            blank_count = 0
            continue

        blank_count += 1
        if blank_count <= 1:
            clean_lines.append(line)

    answer = "\n".join(clean_lines)
    source_list = []

    for source in sources:
        if source and source not in source_list:
            source_list.append(source)

    formatted_sources = "\n".join(
        f"{index}. {source}" for index, source in enumerate(source_list, start=1)
    )

    if not formatted_sources:
        formatted_sources = "No sources found."

    return f"Answer\n------\n{answer}\n\nSources\n-------\n{formatted_sources}"


def chat() -> None:
    db = get_vector_store()
    transcript = SYSTEM_PROMPT

    print("Swift RAG chat. Type /exit to quit or /clear to reset history.")

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question == "/exit":
            break

        if question == "/clear":
            transcript = SYSTEM_PROMPT
            print("History cleared.")
            continue

        results = db.similarity_search_with_score(question, k=RESULT_COUNT)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = (
            f"{transcript}\n\n"
            f"Swift documentation excerpts for the next user question:\n"
            f"{context_text}\n\n"
            f"User: {question}\n"
            f"Assistant:"
        )

        response = ChatOllama(model=CHAT_MODEL).invoke(prompt)
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        answer = response_text(response).strip()

        print()
        print(format_response(answer, sources))

        transcript += f"\n\nUser: {question}\nAssistant: {answer}"


def main() -> None:
    parse_args()
    chat()


if __name__ == "__main__":
    main()
