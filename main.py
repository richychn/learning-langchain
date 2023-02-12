from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm_math.base import LLMMathChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from pathlib import Path
from gpt_index import download_loader, GPTSimpleVectorIndex
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0)

# Initiate data loaders
if os.path.exists('./transactions_index.json'):
    csv_index = GPTSimpleVectorIndex.load_from_disk('transactions_index.json')
else:
    PandasCSVReader = download_loader("PandasCSVReader")
    loader = PandasCSVReader()
    csv_docs = loader.load_data(file=Path('./transactions.csv'))
    csv_index = GPTSimpleVectorIndex(csv_docs)
    csv_index.save_to_disk('transactions_index.json')

print("Data loaded")

# Defne Langchain tools
tools = [
    Tool(
        name="CSV Index",
        func=lambda q: csv_index.query(q),
        description="""
                    Useful when you want to get data about the CSV file you have uploaded. The input to this tool
                    should be full English sentences.
                    """,
    ),
    Tool(
        "Calculator",
        LLMMathChain(llm=llm).run,
                    """
                    Useful for when you need to make any math calculations. Use this tool for any and all numerical calculations. 
                    The input to this tool should be a mathematical expression.
                    """,
    ),
]

# Initialize Langchain agent and chain 
memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True, memory=memory
)

print("Agent loaded")

while True:
    q = input("What would you like to know about the dataset?\n")
    output = agent_chain.run(input=q)
    print(output, "\n")