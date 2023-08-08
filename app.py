# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# set API key for OpenAI
# (sub out here for other LLMs)
os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0, verbose=True)
embeddings = OpenAIEmbeddings()

# create and load PDF loader
loader = PyPDFLoader('ESMT_AIM_Brochure_FINAL.pdf')
# split pages from pdf
pages = loader.load_and_split()
# load documents into vector database aka chromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='brochure')

# create vectorstore info object - metadata repo
vectorstore_info = VectorStoreInfo(
    name="brochure",
    description="a brochure for an esmt program",
    vectorstore=store
)
# convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('PDFbot')

# Create a text input box for the user
prompt = st.text_input('say something')

#  if the user hits enter
if prompt:
    # pass the prompt to the LLM
    # response = llm(prompt)
    # swap out the raw llm for a document agent
    response = agent_executor.run(prompt)
    # ..and write out to the screen
    st.write(response)

    # with a streamlit expander
    with st.expander('Document Similarity Search'):
        # find relevant pages
        search = store.similarity_search_with_score(prompt)
        # write out the first
        st.write(search[0][0].page_content)
