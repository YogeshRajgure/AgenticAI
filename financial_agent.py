from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools # to search on yfanance
from phi.tools.duckduckgo import DuckDuckGo # to search internet
from dotenv import load_dotenv
import os


load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# create a web search agent
web_search_agent = Agent(
    name="Web search agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-specdec", api_key=api_key),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True
)

# create a financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    role="Get financial information",
    model=Groq(id="llama-3.3-70b-specdec", api_key=api_key),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
        ],
    instructions=["Use tables to display the data"],
    show_tools_calls=True,
    markdown=True

)

multi_ai_agent = Agent(
    model=Groq(id="llama-3.3-70b-specdec", api_key=api_key),
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

if __name__ == "__main__":

    multi_ai_agent.print_response("Summarize analyst recommendation anf share the latest news for Nvidia", stream=True)
    # multi_ai_agent.run()
