import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.chains.llm import LLMChain
from langchain.agents import initialize_agent, AgentType

# https://dev.to/timesurgelabs/how-to-make-an-ai-agent-in-10-minutes-with-langchain-3i2n

load_dotenv()

model = ChatAnthropic(model_name="claude")
chain = LLMChain()