
# Document Q&A ChatBot

Taking Attention All You Need & Overview of LLM Model as external data source and establishing a communication with these pdf file. It's an open source LLM Model.

We can use this model at different places like:

- Customer Support: It can help answer frequently asked questions by pulling information directly from product manuals, FAQs, and other support documents.

- Educational Tools: Students and educators can use it to get quick answers from textbooks, research papers, and other academic resources.

- Healthcare: Patients and healthcare providers can use it to access medical records, treatment guidelines, and other healthcare documents.

- Technical Support: It can assist in troubleshooting by providing solutions from technical manuals and user guides.

By leveraging a document Q&A chatbot, you can save time and improve efficiency in accessing and retrieving information from large sets of documents.


# Documentation 

[Conversational RAG](https://python.langchain.com/docs/tutorials/qa_chat_history/)

In many Q&A applications we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of "memory" of past questions and answers, and some logic for incorporating those into its current thinking.

LangChain has integrations with many model providers (OpenAI, Cohere, Hugging Face, etc.) and exposes a standard interface to interact with all of these models.

[Document Loader](https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85)

This is the first piece of the retrieval chain. Langchain provides several options to load external sources of information. Here, we want to load information from a PDF document. So, we will be using the pypdfmodule.


[Text Splitter](https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85)

Once we have loaded the PDF document, the next task is to split the large document into smaller, semantically meaningful chunks to interpret the context and nuances within the document.

For this task, we will be using the RecursiveCharacterTextSplitter. This is the recommended method from LangChain, as it keeps relevant information close together.

[Text Embedding](https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85)

Next, we need to create vector representations of the text chunks. This step makes it feasible to represent text in vector space, allowing for finding texts that are similar to each other. For this experiment, we will be using the OpenAI embedding model.

[Vector Store](https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85)

Now, we will be using the embedding model to transform the text chunks into vectors and then store them in a vector store. Such a database stores the vector embeddings and allows for fast retrieval and similarity search.

There are several options available. One option that is free, open-source, and can run on our local machine is Chroma.

Instead of using chunks_docs, you can also pass the entire document directly. This approach sends the whole document as vectors to the store, potentially improving search results by keeping all information intact. However, this method often exceeds the maximum token limit allowed for a large language model (LLM).


[Retriever](https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85)

The next step would be to retrieve similar information given an unstructured query. For this, we will need a retriever that has the vector store as its backbone from which it will retrieve relevant documents. It takes the query as input and outputs documents that are similar to the query.

[Retrieval Chain](https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85)

Let’s first load the OpenAI chat model. Now, let’s create the retrieval chain connecting the LLM, our query, and the relevant documents. First, we will use the create_stuff_documents_chain function to format the relevant documents into a prompt, and then pass that information to the LLM. We will use a system template that will have a context placeholder to hold the retrieved documents.


Instead of doing this in two steps, i.e., first finding relevant documents using the retriever and then passing the documents to the prompt, we can do it in one step using the create_retriever_chain function. We will start by importing a Q&A prompt template from the LangChain Hub, then create a document chain using the create_stuff_document_chainfunction to chain the LLM and the prompt together, and finally use the create_retriever_chain function to chain the retriever with the document chain.
























 








## Important Libraries Used

 - [ChatMessageHistory](https://python.langchain.com/v0.1/docs/modules/memory/chat_messages/)
 - [BaseChatMessageHistory](https://python.langchain.com/api_reference/core/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html)
- [RunnableWithMessageHistory](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/)
 - [RunnablePassthrough](https://python.langchain.com/v0.1/docs/expression_language/primitives/passthrough/)
 - [PromptTemplate](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/)
- [Retrieval Chain](https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85)
- [Retriever](https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85)
- [HuggingFaceEmbeddings](https://api.python.langchain.com/en/latest/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html)
- [StrOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)

- [Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/)






## Plateform or Providers

 - [langchain_groq](https://python.langchain.com/docs/integrations/chat/groq/)
 - [https://huggingface.co/blog/langchain](https://smith.langchain.com/hub)

## Model

 - LLM - Llama3-8b-8192


## Installation

Install below libraries

```bash
  pip install langchain
  pip install langchain_community
  pip install langchain_groq
  pip install langchain-core
  pip install langchain_huggingface
  pip install Chroma
  pip install Ollama
  pip install bs4


```
    
## Tech Stack

**Client:** Python, LangChain PromptTemplate, ChatGroq,OllamaEmbeddings,FAISS

**Server:** streamlit, PyCharm


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`GROQ_API_KEY`
`HUGGINGFACE_API_KEY`


