#Import all libraries

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from IPython.display import display, Image, Markdown
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Create a state
class State(TypedDict):
    messages : Annotated[list, add_messages]


# Create tools 

@tool
def get_stock_price(symbol: str) -> float:
    '''Return the current price of a stock given the stock symbol
    :param symbol: stock symbol
    :return: current price of the stock
    '''
    return {
        "MSFT": 200.3,
        "AAPL": 100.4,
        "AMZN": 150.0,
        "RIL": 87.6
    }.get(symbol, 0.0)


@tool
def buy_stocks(symbol: str, quantity : int, total : float) -> str:
    """Buy the stocks for the given symbol and quantity and return the total price"""
    decision = interrupt(f"Is it ok to buy {quantity} {symbol} of price {total} ")
    if decision == 'yes':
        return f"I bought {quantity} {symbol} of price {total}"
    else:
        return "Decline"


tools = [get_stock_price,buy_stocks]

# LLM 

llm = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
)
llm_with_tools = llm.bind_tools(tools)

# Create node function

def chatbot_node(state:State) -> State:
    return {'messages':[llm_with_tools.invoke(state['messages'])]}

# Add nodes and edges

builder = StateGraph(State)

builder.add_node('chatbot', chatbot_node)
builder.add_node('tools',ToolNode(tools))

builder.add_edge(START, 'chatbot')
builder.add_conditional_edges('chatbot', tools_condition)
builder.add_edge('tools','chatbot')

# Compile with memory checkpoint
graph = builder.compile(checkpointer=memory)

config = {'configurable': {'thread_id':'1'}}

msg = "I need the total of MSFT and AAPL stock prices"

result = graph.invoke({'messages':[{'role':'user','content':msg}]},config=config)
print(result['messages'][-1].content)

msg = "Now take 10 of RIL and add it to the previous output. And buy it"
result = graph.invoke({'messages':[{'role':'user','content':msg}]},config=config)

decision = input("Approve (yes/no):")

if decision == 'yes':
    result = graph.invoke(Command(resume=decision),config=config)

    
print(result['messages'][-1].content)