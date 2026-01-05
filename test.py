import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

from db import DatabaseClient
from llm import LLMClient
from settings import Settings


settings = Settings()

embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)


# ---------------- LLM HELPERS ---------------- #

def table_description(llm, schema: dict) -> str:
    prompt = f"""
Explain what this database table represents in business terms.

Schema:
{json.dumps(schema, indent=2)}

Return only plain text.
"""
    return llm.invoke(prompt)


def column_descriptions_batch(llm, table_name: str, columns: list) -> list:
    prompt = f"""
You are a database expert.

Explain the business meaning of each column in ONE short sentence.

Table: {table_name}

Columns:
{json.dumps(columns, indent=2)}

Return STRICT JSON ONLY in this format:
[
  {{
    "column": "<column_name>",
    "description": "<text>"
  }}
]

No markdown. No extra text.
"""
    response = llm.invoke(prompt)
    return json.loads(response)


# ---------------- WORKER ---------------- #

def process_table(table_name: str):
    print(f"Processing {table_name}")

    db = DatabaseClient(settings)
    llm = LLMClient(settings)

    schema = db.get_table_schema(table_name)

    # TABLE DOCUMENT
    table_doc = Document(
        id=table_name,
        page_content=table_description(llm, schema),
        metadata=schema
    )

    # COLUMN DOCUMENTS (batched)
    column_docs = []
    schema_columns = {c["column_name"] for c in schema["columns"]}

    col_descs = column_descriptions_batch(
        llm,
        table_name,
        schema["columns"]
    )

    for col in col_descs:
        if col["column"] not in schema_columns:
            continue

        column_docs.append(
            Document(
                id=f"{table_name}.{col['column']}",
                page_content=col["description"],
                metadata={
                    "table": table_name,
                    "column": col["column"]
                }
            )
        )

    return table_doc, column_docs


# ---------------- INGEST ---------------- #

def ingest():
    db = DatabaseClient(settings)
    tables = db.get_tables()

    table_docs = []
    column_docs = []

    with ThreadPoolExecutor(max_workers=settings.INGEST_WORKERS) as executor:
        futures = [executor.submit(process_table, t) for t in tables]

        for future in as_completed(futures):
            table_doc, cols = future.result()
            table_docs.append(table_doc)
            column_docs.extend(cols)

    FAISS.from_documents(table_docs, embeddings)\
         .save_local(settings.FAISS_TABLE_PATH)

    FAISS.from_documents(column_docs, embeddings)\
         .save_local(settings.FAISS_COLUMN_PATH)

    print("✅ Vector stores created (tables + columns)")


if __name__ == "__main__":
    ingest()
