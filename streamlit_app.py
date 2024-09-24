import streamlit as st
from langchain_community.utilities import SQLDatabase
from typing import Any
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Literal
from langchain_core.messages import AIMessage
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
import ast

# Initialize SQLite database connection
db = SQLDatabase.from_uri("sqlite:///hr_database_1.db")

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def handle_tool_error(state) -> dict:
    """
    Handle tool errors and return appropriate error messages.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Initialize Azure OpenAI model
llm = AzureChatOpenAI(
    openai_api_type="azure",
    model_name= "gpt-4o",
    openai_api_version= "2024-02-15-preview",
    azure_endpoint=st.secrets["azure_endpoint"],
    deployment_name=st.secrets["deployment_name"], 
    openai_api_key= st.secrets["openai_api_key"]
)

# Set up SQL database toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Extract specific tools from the toolkit
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result

# Define system prompt for query checking
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

# Create query check prompt and bind it to the LLM
query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check = query_check_prompt | llm.bind_tools(
    [db_query_tool], tool_choice="required"
)

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Initialize the graph workflow
workflow = StateGraph(State)

def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    """
    Initialize the workflow with the first tool call to list database tables.
    """
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if the query is correct before executing it.
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}

# Add nodes to the workflow
workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

# Add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = llm.bind_tools([get_schema_tool])
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)

# Define a tool to submit the final answer
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")

# Define system prompt for query generation
query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

Output the SQL query that answers the input question without a tool call.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

If you get an error while executing a query, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

# Create query generation prompt and bind it to the LLM
query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | llm.bind_tools([SubmitFinalAnswer])

def query_gen_node(state: State):
    """
    Generate a query based on the current state and handle potential errors.
    """
    message = query_gen.invoke(state)

    # Handle cases where the LLM calls the wrong tool
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}

# Add nodes for query generation and execution
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("correct_query", model_check_query)
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    """
    Determine whether to continue the workflow or end it based on the current state.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"

# Define the edges between nodes in the workflow
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges("query_gen", should_continue)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")

# Compile the workflow into a runnable
app = workflow.compile()

# # Sample questions for testing
# questions = [
#     "name 7 the employees who were hired after the end of 2022 with their joining date.",
#     "Get the total salary of all employees in the Engineering department.",
#     "Find the email addresses of employees who receive a benefit called Health Insurance",
#     "List all job titles available in the Human Resources department.",
#     "Find the hire dates of employees who report to manager with manager_id = 536.",
#     "Get the names of employees who receive benefits amounting to more than $1000 and work in the Marketing department.",
#     "List the names and job titles of employees whose total benefit amount exceeds their salary.",
#     "Find all employees who report to the same manager as Dustin Pearson and have benefits under $500.",
#     "Retrieve the names and departments of employees who joined in 2023 and receive more than one benefit.",
#     "Find employees who have the highest total benefit amount and work in Engineering."
# ]

# # Execute the workflow and print the final response
# stream = [x for x in app.stream({"messages": [("user", questions[4])]})]
# agent_final_response = ast.literal_eval(stream[-1]['query_gen']['messages'][0].additional_kwargs['tool_calls'][0]['function']['arguments'])['final_answer']
# print(agent_final_response)









## Streamlit UI for chatbot
# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("SQL Query Bot")
st.write("Ask the bot SQL-related questions and get precise answers!")

# Input field for the user query
user_query = st.text_input("Enter your SQL question:", "")

# When the user submits a query
if user_query:
    # Append the user query to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Execute the workflow
    stream = [x for x in app.stream({"messages": [("user", user_query)]})]
    try:
        agent_final_response = ast.literal_eval(stream[-1]['query_gen']['messages'][0].additional_kwargs['tool_calls'][0]['function']['arguments'])['final_answer']
    except:
        agent_final_response = "Sorry, I couldn't process your query."

    # Append the bot response to the chat history
    st.session_state.chat_history.append({"role": "bot", "content": agent_final_response})

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        st.write(f"**You**: {message['content']}")
    else:
        st.write(f"**Bot**: {message['content']}")

















# # 1.
# ## Streamlit UI for chatbot
# # Initialize chat history in session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Streamlit UI
# st.title("Gen AI Data Analyst")
# # st.write("Ask the bot SQL-related questions and get precise answers!")

# # Input field for the user query
# user_query = st.text_input("Natural Language question:", "")

# # When the user submits a query
# if user_query:
#     # Append the user query to the chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_query})

#     # Use spinner to indicate processing
#     with st.spinner('Processing your query...'):
#         # Execute the workflow
#         try:
#             stream = [x for x in app.stream({"messages": [("user", user_query)]})]
#             agent_final_response = ast.literal_eval(stream[-1]['query_gen']['messages'][0].additional_kwargs['tool_calls'][0]['function']['arguments'])['final_answer']
#         except Exception as e:
#             # agent_final_response = "Sorry, I couldn't process your query."
#             agent_final_response = e

#     # Append the bot response to the chat history
#     st.session_state.chat_history.append({"role": "bot", "content": agent_final_response})

# # Display chat history
# for message in st.session_state.chat_history:
#     if message['role'] == 'user':
#         st.write(f"**User**: {message['content']}")
#     else:
#         st.write(f"**Bot**: {message['content']}\n")


# # 2.
# ## Streamlit UI for chatbot
# # Initialize chat history in session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Initialize user input state
# if 'user_query_input' not in st.session_state:
#     st.session_state.user_query_input = ''

# # Define a function to clear the input field after submission
# def clear_input():
#     st.session_state.user_query_input = ''

# # Streamlit UI
# st.title("SQL Query Bot")
# st.write("Ask the bot SQL-related questions and get precise answers!")

# # Create a container to hold the chat history
# chat_container = st.container()

# # Input field for the user query, tied to session state
# user_query = st.text_input("Enter your SQL question:", st.session_state.user_query_input, key="user_query_input")

# # Submit button to send the message, with a callback to clear the input
# submit = st.button("Send", on_click=clear_input)

# # When the user submits a query
# if submit and user_query:
#     # Append the user query to the chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_query})
    
#     # Add a placeholder for the bot's response with a spinner
#     bot_response_placeholder = st.empty()
    
#     with bot_response_placeholder:
#         with st.spinner('Processing...'):
#             # Execute the workflow
#             stream = [x for x in app.stream({"messages": [("user", user_query)]})]
#             try:
#                 agent_final_response = ast.literal_eval(stream[-1]['query_gen']['messages'][0].additional_kwargs['tool_calls'][0]['function']['arguments'])['final_answer']
#             except:
#                 agent_final_response = "Sorry, I couldn't process your query."

#             # Append the bot response to the chat history
#             st.session_state.chat_history.append({"role": "bot", "content": agent_final_response})

# # Display chat history in a chat-like manner
# with chat_container:
#     for message in st.session_state.chat_history:
#         if message['role'] == 'user':
#             st.markdown(f"**You**: {message['content']}")
#         else:
#             st.markdown(f"**Bot**: {message['content']}")


# # Streamlit app
# st.title("SQL Query Bot")
# st.write("Ask the bot SQL-related questions and get precise answers!")
# # Initialize the conversation history
# conversation_history = []

# # Function to generate the chatbot response
# def generate_response(user_input):
#     with st.spinner("Generating response..."):
#         stream = [x for x in app.stream({"messages": [("user", user_input)]})]
#         agent_final_response = ast.literal_eval(stream[-1]['query_gen']['messages'][0].additional_kwargs['tool_calls'][0]['function']['arguments'])['final_answer']
#         return agent_final_response

# # Main app loop
# while True:
#     user_input = st.text_area("You:", height=100)
#     if st.button("Send"):
#         conversation_history.append("You: " + user_input)
#         response = generate_response(user_input)
#         conversation_history.append("Chatbot: " + response)
#         st.write("\n".join(conversation_history))