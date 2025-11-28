from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

db = SQLDatabase.from_uri("sqlite:////Users/abhijitsen/Downloads/text-to-sql/Chinook.db", sample_rows_in_table_info=0)

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    print(f'Query being run: {query} \n\n')
    return db.run(query)

print(get_schema(''))

def get_llm():
    # Using HuggingFace open source model instead of ChatOpenAI
    llm = HuggingFaceEndpoint(
        # repo_id="Qwen/Qwen1.5-7B-Chat",
        repo_id="google/flan-t5-small",
        task="text-generation",
        # provider="huggingface",  # or your configured provider
    )
    return ChatHuggingFace(llm=llm)

def write_sql_query(llm):
    template = """Based on the table schema below, write a SQL query that would answer the user's question:
    {schema}

    Question: {question}
    SQL Query:"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given an input question, convert it to a SQL query. No pre-amble. "
            "Please do not return anything else apart from the SQL query, no prefix aur suffix quotes, no sql keyword, nothing please"),
            ("human", template),
        ]
    )

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def answer_user_query(query, llm):
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""

    prompt_response = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
            ),
            ("human", template),
        ]
    )

    full_chain = (
        RunnablePassthrough.assign(query=write_sql_query(llm))
        | RunnablePassthrough.assign(
            schema=get_schema,
            response=lambda x: run_query(x["query"]),
        )
        | prompt_response
        | llm
    )

    return full_chain.invoke({"question": query})

load_dotenv()

query = 'Give some Tracks by the Artist name Audioslave'
response = answer_user_query(query, llm=get_llm())
print(response.content)