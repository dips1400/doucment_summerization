import os
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from flask import Flask, jsonify, request
from flask_cors import CORS


import os, tempfile, sys
from io import BytesIO
from io import StringIO
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain import LLMChain


from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


# Load environment variables from .env file
load_dotenv()


app = Flask(__name__)
CORS(app)

# Read OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = "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"

@app.route('/hello', methods=['GET', 'POST'])
def home():
    return "Hello World from Dipti"

@app.route('/data', methods=['GET', 'POST'])
def my_data():
    
        file = request.files['file']
        # print(file)
        # return "done"
        
        # file = request.files['file']
        # print(file.read())

        file.save(file.filename)

        MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]
        max_tokens = {"gpt-4":7000, "gpt-4-32k":31000, "gpt-3.5-turbo":3000}
        TEMPERATURE_MIN_VALUE = 0.0
        TEMPERATURE_MAX_VALUE = 1.0
        TEMPERATURE_DEFAULT_VALUE = 0.9
        TEMPERATURE_STEP = 0.01

        temperature=0
        # model_name = model_name(label="Model", options=MODEL_OPTIONS)

        # agent = create_csv_agent(OpenAI(api_key="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", temperature=0), file.filename, verbose=True)

        # print(agent.agent.llm_chain.prompt.template)

        if file is not None:

            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=0)
            agent = create_csv_agent(OpenAI(api_key="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", temperature=0), file.filename, verbose=True)
            loader = CSVLoader(agent)
            data = loader.load()
            # texts = text_splitter.split_documents(data)

            # try:
            #     loader = CSVLoader(file=tmp_file, encoding="cp1252")
            #     #loader = UnstructuredFileLoader(tmp_file_path)
            #     data = loader.load()
            #     texts = text_splitter.split_documents(data)
            # except:
            #     # loader = CSVLoader(file=tmp_file, encoding="utf-8")
            #     agent = create_csv_agent(OpenAI(api_key="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", temperature=0), file.filename, verbose=True)
            #     # #loader = UnstructuredFileLoader(tmp_file_path)
            #     # data = loader.load()
            #     texts = text_splitter.split_documents(agent)


            # Initialize the OpenAI module, load and run the summarize chain
            # llm = OpenAI(model_name=model_name, temperature=temperature)
            llm = OpenAI(temperature=temperature)
            chain = load_summarize_chain(llm, chain_type="stuff")
            #search = docsearch.similarity_search(" ")
            summary = chain.run(data)


        os.remove(file.filename)

        return jsonify({summary})




if __name__ == '__main__':
  app.run(debug=True)