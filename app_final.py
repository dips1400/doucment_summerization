import openai
import os
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

# Load environment variables from .env file
load_dotenv()


app = Flask(__name__)
CORS(app)

# Read OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = "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"


@app.route('/getdata', methods=['GET', 'POST'])
def getdata():

    # Set your OpenAI API key
    openai.api_key = "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"

    # Load CSV file into a DataFrame
    df = pd.read_csv('E:/NPL_Project/document_summarization/Backend/text_segments.csv')

    #Combine all text columns into one
    # text_column = ' '.join(df['text'])
    text_column = (df["text"])

    # Define the prompt for GPT-3 summarization
    prompt = f"Summarize the following text: {text_column}"

    # Call the OpenAI API for text summarization
    # response = openai.Completion.create(
    # engine="text-davinci-003",  # Choose the GPT-3 engine
    # prompt=prompt,
    # max_tokens=150  # Adjust as needed
    # )
   
    response = openai.completions.create(model="gpt-3.5-turbo", prompt="summarize the file",max_tokens=150)

    # Extract the generated summary from the API response
    generated_summary = response.choices[0].text.strip()

    # Print or use the generated summary
    print(generated_summary)
    return jsonify(generated_summary)




if __name__ == '__main__':
  app.run(debug=True)