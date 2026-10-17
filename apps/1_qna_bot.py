from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
st.set_page_config(layout="wide")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# while True:
#     question = input("User: ")
#     if question.lower() in ['exit', 'quit', 'q', 'bye']:
#         print("Exiting the chat. Goodbye!")
#         break

#     result = llm.invoke(question)
#     print("AI :",result.text, "\n")

st.title("Q&A AI Bot")
st.markdown("QnA bot with LangChain and Google Gemini!")

if "messages" not in st.session_state:
    st.session_state.messages =[]

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything:")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    result = llm.invoke(st.session_state.messages)
    st.chat_message("ai").markdown(result.text)
    st.session_state.messages.append({"role": "ai", "content": result.text})



# streamlit run .\1_qna_bot.py