# my_ai_agent.py

# Your existing imports
from dotenv import load_dotenv, find_dotenv
import os
import base64
import io
import json
import operator
from functools import partial
from typing import Annotated, List, Literal, Optional, Sequence, TypedDict

import pandas as pd
from IPython.display import display
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from matplotlib.pyplot import imshow
from langsmith import traceable
from datetime import datetime, date
from PIL import Image
import requests







# Load environment variables from .env file
load_dotenv()

# Access the environment variables (ensure all variables are properly set)
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_API_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
SESSIONS_POOL_MANAGEMENT_ENDPOINT = os.getenv('SESSIONS_POOL_MANAGEMENT_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_VERSION')
SQL_SERVER = os.getenv('SQL_SERVER')
SQL_USER = os.getenv('SQL_USER')
SQL_PWD = os.getenv('SQL_PWD')
SQL_DATABASE = os.getenv('SQL_DATABASE')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_API_VERSION
)

# Initialize the Sessions Python REPL Tool
repl = SessionsPythonREPLTool(
    pool_management_endpoint=SESSIONS_POOL_MANAGEMENT_ENDPOINT
)

# Define AgentState and RawToolMessage
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class RawToolMessage(ToolMessage):
    raw: dict
    tool_name: str

# Define your tools (create_df_from_sql and python_shell)
class create_df_from_sql(BaseModel):
    """Execute a SQL SELECT statement and use the results to create a DataFrame with the given column names."""
    select_query: str = Field(..., description="A SQL SELECT statement.")
    df_columns: List[str] = Field(..., description="Ordered names to give the DataFrame columns.")
    df_name: str = Field(..., description="The name to give the DataFrame variable in downstream code.")

class python_shell(BaseModel):
    """Execute Python code that analyzes the DataFrames that have been generated. Make sure to print any important results."""
    code: str = Field(
        ...,
        description="The code to execute. Make sure to print any important results.",
    )

# Replace this with your actual schema
schema = {}
# Fetch the schema
url = "http://roapi-app-arash.dhbjd9bvccbjedea.eastus.azurecontainer.io:8000/api/schema"
response = requests.get(url)
if response.status_code == 200:
    schema = response.json()
    print("Schema fetched successfully:")
else:
    print(f"Failed to fetch schema. Status code: {response.status_code}")
    schema = {}

from sqlalchemy import Table, MetaData, Column, Integer, String, Boolean

def map_data_type(data_type):
    if data_type == "Utf8":
        return String
    elif data_type == "Int64":
        return Integer
    elif data_type == "Int32":
        return Integer
    elif data_type == "Boolean":
        return Boolean
    else:
        return String  # Default to String if unknown

def convert_json_to_table_info(json_output):
    metadata = MetaData()
    tables = []

    for table_name, table_info in json_output.items():
        columns = []
        for field in table_info["fields"]:
            col_type = map_data_type(field["data_type"])
            col_args = {"nullable": field.get("nullable", True)}
            columns.append(Column(field["name"], col_type, **col_args))
        table = Table(table_name, metadata, *columns)
        tables.append(table)
    return tables

try:
    tables = convert_json_to_table_info(schema)
    print("Tables parsed successfully:")
    for table in tables:
        print(table)
except Exception as e:
    print(f"Error parsing schema: {e}")
    tables = []

def generate_table_descriptions(tables):
    table_descriptions = ''
    for table in tables:
        columns = [f"{column.name} ({str(column.type)})" for column in table.columns]
        table_descriptions += f"Table '{table.name}' with columns: {', '.join(columns)}\n"
    return table_descriptions

table_descriptions = generate_table_descriptions(tables)
# Date handling functions
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
today = date.today()
formatted_date = today.strftime("%B %d, %Y")

# System prompt
system_prompt = f"""\
You are an expert at ROAPI and Python. You have access to ROAPI \
with the following tables

{table_descriptions}

Given a user question related to the data in the database, \
first generate the SQL query that conforms to ApacheArrow and Datafusion style. Pay attention to the columns that need to be in double quotes and the ones that should not be. 
please note the following points:
1- channel is an alias to device_class so if user asks for count of channel you need to do the count of device_class
2- is_live indicates whether the video is live or not so use this if you want to calcualte on demand versus live
3- if you need to compare numbers such as duration or anything else if the results are all zero you should not rank them
Below are some examples of the SQL queries:
1- SELECT COUNT(DISTINCT user_id) AS active_user_count FROM bitmovin WHERE date >= to_date(cast(now() AS VARCHAR)) - INTERVAL '200 days' AND user_id IS NOT NULL;
2- SELECT * FROM bitmovin WHERE date >= to_date(cast(now() AS VARCHAR)) - INTERVAL '200 days' AND user_id IS NOT NULL LIMIT 10;
3- SELECT "AccountType", "DeviceType", "ApiName", COUNT(*) AS api_usage_count FROM asl GROUP BY 
"AccountType", "DeviceType", "ApiName" ORDER BY "AccountType", "DeviceType", api_usage_count DESC;
4- SELECT asl."TenantId", asl."DeviceType", bitmovin."browser", COUNT(*) AS usage_count, AVG(bitmovin."page_load_time") AS avg_page_load_time,
    AVG(bitmovin."startuptime") AS avg_startup_time FROM asl JOIN bitmovin ON asl."TenantId" = bitmovin."tenant"
    GROUP BY asl."TenantId", asl."DeviceType", bitmovin."browser" ORDER BY usage_count DESC LIMIT 10;

Make sure to make date queries align to today's date {formatted_date}
Then tell me the SQL query that you will use. Then get the relevant data from the table as a DataFrame using the create_df_from_sql tool. Then use the \
python_shell to do any analysis required to answer the user question."""
#print(system_prompt)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ]
)

# Define your agent functions
def call_model(state: AgentState) -> dict:
    """Call model with tools passed in."""
    messages = []

    chain = prompt | llm.bind_tools([create_df_from_sql, python_shell])
    messages.append(chain.invoke({"messages": state["messages"]}))

    return {"messages": messages}

def execute_sql_query(state: AgentState) -> dict:
    """Execute the latest SQL queries."""
    messages = []

    for tool_call in state["messages"][-1].tool_calls:
        if tool_call["name"] != "create_df_from_sql":
            continue

        # Execute SQL query
        sql_query = tool_call["args"]["select_query"]
        url = "http://roapi-app-arash.dhbjd9bvccbjedea.eastus.azurecontainer.io:8000/api/sql"
        response = requests.post(url, data=sql_query)

        data_sql = None  # Initialize data_sql

        # Check if the request was successful
        if response.status_code == 200:
            try:
                data_sql = response.json()
                print(data_sql)
            except requests.JSONDecodeError:
                print("Failed to decode JSON response.")
                data_sql = []  # Assign an empty list to avoid further errors
        else:
            print(f"Failed to execute SQL query. Status code: {response.status_code}")
            data_sql = []  # Assign an empty list or handle accordingly

        # Proceed only if data_sql has a valid value
        if data_sql is not None:
            # Convert result to Pandas DataFrame
            df_columns = tool_call["args"]["df_columns"]
            df = pd.DataFrame(data_sql, columns=df_columns)
            df_name = tool_call["args"]["df_name"]

            # Add tool output message
            messages.append(
                RawToolMessage(
                    f"Generated dataframe {df_name} with columns {df_columns}",
                    raw={df_name: df},
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                )
            )
        else:
            # Handle the case where data_sql is None
            error_message = f"Failed to execute SQL query for {tool_call['name']}."
            print(error_message)
            # Optionally, create an empty DataFrame or skip this tool call
            df_columns = tool_call["args"]["df_columns"]
            df = pd.DataFrame([], columns=df_columns)
            df_name = tool_call["args"]["df_name"]

            # Add tool output message with error information
            messages.append(
                RawToolMessage(
                    f"Failed to generate dataframe {df_name}. {error_message}",
                    raw={df_name: df},
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                )
            )

    return {"messages": messages}

def _upload_dfs_to_repl(state: AgentState) -> str:
    """Upload generated dataframes to code interpreter and return code for loading them."""
    df_dicts = [
        msg.raw
        for msg in state["messages"]
        if isinstance(msg, RawToolMessage) and msg.tool_name == "create_df_from_sql"
    ]
    name_df_map = {name: df for df_dict in df_dicts for name, df in df_dict.items()}

    # Data should be uploaded as a BinaryIO.
    # Files will be uploaded to the "/mnt/data/" directory on the container.
    for name, df in name_df_map.items():
        buffer = io.StringIO()
        df.to_csv(buffer)
        buffer.seek(0)
        repl.upload_file(data=buffer, remote_file_path=name + ".csv")

    # Code for loading the uploaded files.
    df_code = "import pandas as pd\n" + "\n".join(
        f"{name} = pd.read_csv('/mnt/data/{name}.csv')" for name in name_df_map
    )
    return df_code

def _repl_result_to_msg_content(repl_result: dict) -> str:
    """Convert REPL results to message content."""
    content = {}
    for k, v in repl_result.items():
        if isinstance(repl_result[k], dict) and repl_result[k]["type"] == "image":
            base64_str = repl_result[k]["base64_data"]
            img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
            display(img)
        else:
            content[k] = repl_result[k]
    return json.dumps(content, indent=2)

def execute_python(state: AgentState) -> dict:
    """Execute the latest generated Python code."""
    messages = []

    df_code = _upload_dfs_to_repl(state)
    last_ai_msg = [msg for msg in state["messages"] if isinstance(msg, AIMessage)][-1]
    for tool_call in last_ai_msg.tool_calls:
        if tool_call["name"] != "python_shell":
            continue

        generated_code = tool_call["args"]["code"]
        repl_result = repl.execute(df_code + "\n" + generated_code)

        messages.append(
            RawToolMessage(
                _repl_result_to_msg_content(repl_result),
                raw=repl_result,
                tool_call_id=tool_call["id"],
                tool_name=tool_call["name"],
            )
        )
    return {"messages": messages}

def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue processing."""
    return "execute_sql_query" if state["messages"][-1].tool_calls else END

# Build the workflow
workflow = StateGraph(AgentState)

workflow.add_node("call_model", call_model)
workflow.add_node("execute_sql_query", execute_sql_query)
workflow.add_node("execute_python", execute_python)

workflow.set_entry_point("call_model")
workflow.add_edge("execute_sql_query", "execute_python")
workflow.add_edge("execute_python", "call_model")
workflow.add_conditional_edges("call_model", should_continue)

# Compile the app
app = workflow.compile()
# Add logging statements
