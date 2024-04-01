import os
from requests.exceptions import ConnectionError
from flask import Flask, request, Response, stream_with_context
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
import pinecone
import json

from langchain.chat_models import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import (
	LLMChain,
	ConversationalRetrievalChain,
)
from langchain.schema.runnable import RunnableSequence, Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import (
	ChatPromptTemplate,
	SystemMessagePromptTemplate,
	HumanMessagePromptTemplate,
	MessagesPlaceholder,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.environ.get("PINECONE_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

app = Flask(__name__)

# this will need to be reconfigured before taking the app to production
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/chat", methods=["POST"])
@cross_origin()
def chat():
	try:
		body = request.json

		template = """You are conversational chatbot assistant. {prefix} Answer base on the only given theme. Start a 
		natural seeming conversation about anything that relates to the lessons content. Given the following conversation 
		and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
			Chat History: {chat_history}
			Follow Up Input: {question}
			Standalone question:"""

		prompt = PromptTemplate.from_template(template)

		llm = ChatOpenAI(
			openai_api_key=OPENAI_API_KEY,
			model_name=OPENAI_MODEL_NAME,
			streaming=True,
			callbacks=[StreamingStdOutCallbackHandler()],
		)

		# llm = GoogleGenerativeAI(model="models/gemini-pro", google_api_key=GOOGLE_API_KEY)

		memory = ConversationSummaryBufferMemory(
			memory_key="chat_history",
			llm=llm,
			# max_token_limit=300,
			return_messages=True,
			input_key="question",
		)

		pinecone.init(api_key=PINECONE_KEY, environment=PINECONE_ENV)
		embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

		vectorstore = Pinecone.from_existing_index(
			index_name=PINECONE_INDEX,
			embedding=embeddings,
			namespace=PINECONE_NAMESPACE,
		)
		retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

		qa = ConversationalRetrievalChain.from_llm(
			llm=llm,
			retriever=retriever,
			condense_question_prompt=prompt,
			memory=memory,
			# callbacks=[StreamingStdOutCallbackHandler()],
		)
		response = qa(
			{
				"prefix": body["system_input"],
				"chat_history": "",
				"question": body["human_input"],
			}
		)

		return response["answer"]
	except Exception as e:
		return {"error": "{}: {}".format(type(e).__name__, str(e))}, 500


@app.route("/feedback", methods=["POST"])
@cross_origin()
def feedback():
	body = request.json

	template = """
        Chat History:
        {chat_history}

        Follow Up Input: {query}
        Analyze:"""

	prompt = PromptTemplate.from_template(template)

	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
	for item in body["chat_history"]:
		if item["who"] == "ai":
			memory.chat_memory.add_ai_message(item["text"])
		else:
			memory.chat_memory.add_user_message(item["text"])

	llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)

	conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
	response = conversation(
		{
			"query": "Analyzes the following conversation and provides possible feedback, such as grammar or spelling errors, all about the HumanMessage. Respond using markdown."
		}
	)

	return response["text"]


@app.route("/")
def hello_world():  # put application's code here
	return "Hello World!"


if __name__ == "__main__":
	app.run()
