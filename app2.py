from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain.document_loaders.csv_loader import CSVLoader
from fileinput import filename
import pandas as pd
from langchain.agents import AgentExecutor

# prompt = PromptTemplate.from_template(prompt_template)
load_dotenv()


app = Flask(__name__)
CORS(app)

# Read OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = "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"


@app.route('/datass', methods=['GET', 'POST'])
def my_data():
    
        file = request.files['file']
        # print(file)
        # return "done"
        
        # file = request.files['file']
        # print(file.read())

        file.save(file.filename)

        loader = create_csv_agent(OpenAI(api_key="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", temperature=0), file.filename, verbose=True)

        # loader = CSVLoader(file_path="/document_summerization/Backend/text_segment.csv", encoding="utf-8")
        # loader = pd.read_csv(created_csv,encoding='unicode_escape',delim_whitespace=True)
        # docs = loader.load()
        # print(docs)

        # print(agent.agent.llm_chain.prompt.template)
        # Define prompt
        # prompt_template = """Write a concise summary of the following:
        # "{text}"
        # CONCISE SUMMARY:"""
        # prompt = PromptTemplate.from_template(prompt_template)

        llm = OpenAI(api_key="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", temperature=0)

        summaries = []
        
        loader = CSVLoader(file)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        print("Summary for: ", file)
        print(summary)
        print("\n")
        summaries.append(summary)
            
        return summaries

       
        # llm_chain = LLMChain(llm=llm,prompt=prompt)

        # # Define StuffDocumentsChain
        # stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        
        # summary = stuff_chain.run(docs)

        os.remove(file.filename)

        return jsonify(summary)

if __name__ == '__main__':
  app.run(debug=True)