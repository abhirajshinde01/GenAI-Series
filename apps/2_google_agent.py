from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model = "openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a helpful assistant that uses Google Search to answer questions.",
    checkpointer=memory
)

while True:
    query = input("User: ")
    if query.lower() in ['exit', 'quit', 'q', 'bye']:
        print("Exiting the chat. Goodbye!")
        break

    response = agent.invoke({"messages": [{"role": "user", "content": query}]},
                             {"configurable": {"thread_id": "user1"}})
    
    print("AI :",response["messages"][-1].text, "\n")