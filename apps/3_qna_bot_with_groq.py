from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model = "openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that uses Google Search as well to answer questions.",
    checkpointer=st.session_state.memory
)


#### Building web interface
st.subheader("Quick ANswer - Answers at speed of thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything:")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    response = agent.stream({"messages": [{"role": "user", "content": query}]},
                             {"configurable": {"thread_id": "user1"}},
                             stream_mode="messages")

    ## without streaming use agent.invoke() instead of agent.stream()
    # answer = response["messages"][-1].text
    # st.chat_message("ai").markdown(answer)
    #st.session_state.history.append({"role": "ai", "content": answer})

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()
        message = ""

        for chunk in response:
            message += chunk[0].content
            space.write(message)

        st.session_state.history.append({"role": "ai", "content": message})


