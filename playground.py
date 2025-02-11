from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools # to search on yfanance
from phi.tools.duckduckgo import DuckDuckGo # to search internet
import phi.api
import phi
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
import os
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")
groq_api=os.getenv("GROQ_API_KEY")

Groq.api_key=groq_api

web_search_agent = Agent(
    name="Web search agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-specdec"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True
)

# create a financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    role="Get financial information",
    model=Groq(id="llama-3.3-70b-specdec"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
        ],
    instructions=["Use tables to display the data"],
    show_tools_calls=True,
    markdown=True

)


app = Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
