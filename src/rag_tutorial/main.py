import argparse
import os
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
CHROMA_PATH = "chroma"
DOCS_PATH = "swift-book-documentation"
DOCS_GLOB = "**/*.md"
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"
RESULT_COUNT = 4

PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}

    ---

    Answer the question based on the above context: {question}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("question_text", type=str, help="Ask me anything about Swift.")
    return parser.parse_args()


def get_vector_store() -> Chroma:
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_PATH):
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function,
        )

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

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH,
    )


def answer_question(query_text: str) -> tuple[Any, list[str | None]]:
    db = get_vector_store()
    results = db.similarity_search_with_relevance_scores(
        query_text,
        k=RESULT_COUNT,
    )

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(model=CHAT_MODEL)
    response = model.invoke(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    return response, sources


def format_response(response: Any, sources: Iterable[str | None]) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        content = "\n".join(parts)

    lines = [line.rstrip() for line in str(content).strip().splitlines()]
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


def main() -> None:
    args = parse_args()
    response, sources = answer_question(args.question_text)
    print(format_response(response, sources))


if __name__ == "__main__":
    main()
