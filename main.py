import os
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.chains.llm import LLMChain
from langchain.agents import AgentExecutor, create_react_agent
from datetime import datetime

load_dotenv()

# --- 1. Search ---
search = DuckDuckGoSearchRun(verbose=True)


# --- 2. Scrape ---
def scrape_url(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
    }
    req = Request(url=url, headers=headers)
    with urlopen(req) as response:
        response_content = response.read()
        soup = BeautifulSoup(response_content, 'html.parser')
        text_content = soup.get_text()
        return text_content

web_fetch_tool = Tool.from_function(
    func=scrape_url,
    name='WebFetch',
    description='Fetches the content of a web page for a given URL'
)


# --- 3. Summarize ---
llm = ChatAnthropic(
    model='claude-3-haiku-20240307',
    temperature=0.0,
    anthropic_api_key=os.getenv('ANTHROPIC_KEY')
)
chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("Summarize the following content: {content}")
)
summarize_tool = Tool.from_function(
    func=chain.run,
    name="Summarizer",
    description="Summarizes the content of a web page"
)


# --- Setup Agent ---
tools = [search, web_fetch_tool, summarize_tool]
agent_prompt = '''<task>Answer the User Query as best you can.</task>

<information>
You have access to the following tools:
{tools}

Today's date is:
{today}

Use the following format:

Query: The input query you're trying to answer.
Thought: Your thoughts on the next best possible action to take.
Action: The action to take which should be one of [{tool_names}] or answer.
Action Input: The input to the action.
Observation: Your observation based on the action.
... (this Thought/Action/Action Input/Observation can repeat up to 3 times maximum)
Final Answer: The final answer to the original query.
</information>

<user_query>
{input}
</user_query>

<thought>
{agent_scratchpad}
</thought>
'''

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=PromptTemplate.from_template(agent_prompt)
)
executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

result = executor.invoke({
    "input": "What is the latest news on Bitcoin?",
    "today": datetime.now().date()
})
print(result)