# Note issue:
# 1. can not use google_search for sub_agents, if there is no another choice, then use AgentTool

# Flow
# ````
# root_agent
#   |-- animal_agent
#   |   |-- cat_agent
#   |      |-- search_agent (openai/gpt-4o-mini) -> adk_tavily_tool
#   |-- weather_agent -> adk_tavily_tool
#   |-- greeting_agent
#   |-- farewell_agent
# ````

from typing import Dict, TypedDict, Literal
from google.adk.agents import LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import google_search, agent_tool

from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import TavilySearchResults

# Instantiate the LangChain tool
tavily_tool_instance = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# Wrap it with LangchainTool for ADK
adk_tavily_tool = LangchainTool(tool=tavily_tool_instance)

search_agent = LlmAgent(
    model=LiteLlm(model="openai/gpt-4o-mini"),
    name="search_agent", # Requdired: Unique agent name
    description="A helpful assistant agent that can answer questions. Search internet for the given query",
    instruction="""Agent to answer questions using TavilySearch""",
    tools=[adk_tavily_tool], # Provide an instance of the tool
)

def get_weather(city: str) -> Dict[str, str]:
    # Best Practice: Log tool execution for easier debugging
    print(f"--- Tool: get_weather called for city: {city} ---")
    city_normalized = city.lower().replace(" ", "") # Basic input normalization

    # Mock weather data for simplicity (matching Step 1 structure)
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
        "chicago": {"status": "success", "report": "The weather in Chicago is sunny with a temperature of 25°C."},
        "toronto": {"status": "success", "report": "It's partly cloudy in Toronto with a temperature of 30°C."},
        "chennai": {"status": "success", "report": "It's rainy in Chennai with a temperature of 15°C."},
    }

    # Best Practice: Handle potential errors gracefully within the tool
    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "report": f"Sorry, I don't have weather information for '{city}'."}

greeting_agent = LlmAgent(
         model="gemini-2.0-flash-exp",
            name="greeting_agent",
            instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. ",
            description="Handles simple greetings and hellos",
 )

farewell_agent = LlmAgent(
          model="gemini-2.0-flash-exp",
            name="farewell_agent",
            instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message.",
            description="Handles simple farewells and goodbyes",
 )

cat_agent = LlmAgent(
          model="gemini-2.0-flash-exp",
            name="cat_agent",
            instruction="You are the Cat Agent. Your ONLY task is to provide the cat information for the given cat. - Delegation Rules: If cat needs go to hospital, delegate to search_agent to find the nearest hospital",
            description="Handles cat information",
            sub_agents=[search_agent]
 )

animal_agent = LlmAgent(
          model="gemini-2.0-flash-exp",
            name="animal_agent",
            instruction="You are the Animal Agent. Your ONLY task is to provide the animal information for the given animal. - Delegation Rules: About Cat, delegate to cat_agent.Do not perform any other actions.",
            description="Any question about animal such as cat, dog, etc.",
            sub_agents=[cat_agent]
 )

weather_agent = LlmAgent(
          model="gemini-2.0-flash-exp",
            name="weather_agent",
            instruction="You are the Weather Agent. Your ONLY task is to provide the weather information for the given city. I can answer your questions by searching the internet",
            description="Handles weather information",
            tools=[adk_tavily_tool]
 )

root_agent = Agent(
        name="root_agent",
        model="gemini-2.0-flash-exp",
        description="You are the main Agent, coordinating a team. - Your main task: answer the user's query based on your knwoledge. - Delegation Rules: - If the user gives a simple greeting (like 'Hi', 'Hello'), delegate to `greeting_agent`. - If the user gives a simple farewell (like 'Bye', 'See you'), delegate to `farewell_agent`. - If the user gives a weather query (like 'What's the weather in Chennai?'), delegate to `weather_agent`. - If the user gives a animal query (like 'What's the cat information?'), delegate to `animal_agent`. - If the user gives a question related to USD, delegate to `search_agent`.",
        sub_agents=[greeting_agent, farewell_agent, weather_agent, animal_agent]
)
