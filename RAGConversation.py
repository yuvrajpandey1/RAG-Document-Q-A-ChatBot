## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

groq_api_key = "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"
llm= ChatGroq(model_name="Llama3-8b-8192",groq_api_key=groq_api_key)
llm

hf_api_key = "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"
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set up Streamlit
import streamlit as st

st.title("Conversation RAG With PDF uploads and Chat History")
st.write("Upload PDF and Chat with their Content")

# Input the Groq API KEY
api_keys=st.text_input("Enter your Groq API Key:",type="password")

## Check if Groq API KEY is provided
if api_keys:
  llm= ChatGroq(model_name="Gemma2-9b-It",groq_api_key=groq_api_key)
   ## CHat Interface
  session_id=st.text_input("Session ID",value="default_session")
  ## Statefully manage chat history
  if 'store' not in st.session_state:
    st.session_state.store={}

  uploaded_files=st.file_uploader("Choose A PDF file",type="pdf",accept_multiple_files=False)
  ## Process my uploaded PDF's

  if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
      temppdf=f"./temp.pdf"
      with open(temppdf,"wb") as file:
        file.write(uploaded_file.getvalue())
        file_name=uploaded_file.name

      loader=PyPDFLoader(temppdf)
      docs=loader.load()
      documents.extend(docs)

  #Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
    retriever = vectorstore.as_retriever()

  contextualize_q_system_prompt=(
     "Given a chat history and the latest user question"
     "which might reference context in the chat history"
     "formulate a standalone question which can be understood"
     "without the chat history. Do NOT answer the question,"
     "just reformulate it if needed and otherwise return it as is."
  )
  contextualize_q_system_prompt=ChatPromptTemplate.from_messages(
     [
         ("system",contextualize_q_system_prompt),
         MessagePlaceholder("chat_history"),
         ("human","{input}"),
     ]
  )
  history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_system_prompt)

  #Answer Question

  system_prompt=(
      "You are an assistant for question-answering tasks."
      "Use the following pieces of retrieved context to answer"
      "the question. If you don't know the answer, say that you"
      "answer concise"
      "\n\n"
      "{context}"
  )
  qa_prompt = ChatPromptTempate.from_messages(
      [
          ("system",system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human","{input}"),

      ]
  )
  question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
  rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

  def get_session_history(session:str)-> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
      st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]

  conversational_rag_chain=RunnableWithMessageHistory(
      rag_chain,get_session_history,
      input_messages_key="input",
      history_messages_key="chat_history",
      output_messages_key="answer"
  )

  user_input = st.text_input("Your question:")
  if user_input:
    session_history=get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input":user_input},
        config={
            "configurable":{"session_id":session_id}
        } # constructs a key "abc123" in 'store'.
    )
    st.write(st.session_state.store)
    st.write("Assistant:",response['answer'])
    st.write("Chat History:",session_history.messages)

else:
  st.warning("Please enter the Groq API KEY")

