import logging
import os
from typing import Dict, List, TypedDict, Literal

import requests
from dotenv import find_dotenv, load_dotenv
from geopy.geocoders import Nominatim
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, StateGraph
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger()
# model defination for agentic use cases
load_dotenv(find_dotenv())

MODEL = ChatNVIDIA(
    model="moonshotai/kimi-k2-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=1,
    top_p=0.9,
    max_completion_tokens=16384,
)
class AgentState(TypedDict):
    messages: List[Dict]
    user_input: str
    goal: str
    plan: List[str]
    current_action: str
    next_action: str
    tool_result: Dict
    tools_used: List[Dict]
    working_memory: Dict
    long_term_memory: Dict
    observations: List[Dict]
    last_error: str
    retry_count: int
    confidence: float
    status: str
    final_response: str
    source : dict

# pip command to install langchain_nvidia_ai_endpoints
class Tools:
    def __init__(self):
        TAVILY_MAX_RESULTS = 1
        self.SEARCH_ENGINE = TavilySearch(
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            max_results=TAVILY_MAX_RESULTS,
            include_answer=True,
            include_raw_content=True,
            include_urls=True,
            include_tables=True,
            include_domains=[
                "help.sap.com",
                "www.sap.com",
                "developers.sap.com",
                "api.sap.com",
                "community.sap.com",
            ],
            include_images=True,
        )
        self.geolocator = Nominatim(user_agent="my_agent")

    def weather_tool(self, city: str, country: str = None, days: int = 1) -> Dict:
        """
        Get current weather and forecast information for a given city.
        Use this tool when the user asks about:
        - weather
        - temperature
        - humidity
        - rain
        - forecast
        - climate conditions
        Args:
            city (str): Name of the city.
            country (str, optional): Country name or code.
            days (int, optional): Number of forecast days.
        Returns:
            Dict containing:
            - location
            - coordinates
            - current weather
            - forecast data
        """
        try:
            location_query = city
            if country:
                location_query += f", {country}"
            location = self.geolocator.geocode(location_query)
            if not location:
                return {"error": "Location not found"}
            latitude = location.latitude
            longitude = location.longitude
            # Weather API
            api_url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "timezone": "auto",
                "forecast_days": days,
            }
            response = requests.get(api_url, params=params)
            data = response.json()
            current = data.get("current", {})
            daily = data.get("daily", {})

            forecast = []
            if daily:
                for i in range(len(daily.get("time", []))):
                    forecast.append(
                        {
                        "date": daily["time"][i],
                        "max_temp": daily["temperature_2m_max"][i],
                        "min_temp": daily["temperature_2m_min"][i],
                        "weather_code": daily["weather_code"][i],
                        }
                    )
            return {
                "location": location_query,
                "latitude": latitude,
                "longitude": longitude,
                "current": {
                    "temp": current.get("temperature_2m"),
                    "humidity": current.get("relative_humidity_2m"),
                    "weather_code": current.get("weather_code"),
                },
                "forecast": forecast,
            }
        except Exception as e:
            return {"error": str(e)}

    def web_search(self, query: str) -> List[Dict]:
        """
        Search the web for real-time information on a given topic.
        Use this tool when:
        - up-to-date information is needed
        - the answer requires internet search
        - the user asks about recent events or external knowledge
        Args:
            query (str): Search query.
        Returns:
            List of search results with relevant web information.
        """
        try:
            response: Dict = self.SEARCH_ENGINE.invoke(query)
            normalized_results = []
            for item in response.get("results", []):
                normalized_results.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                    "score": item.get("score"),
                })
            return {
                "query": query,
                "summary": response.get("answer", ""),
                "images": response.get("images", []),
                "sources": normalized_results,
            }
        except Exception as e:
            return {
                "error": str(e)
            }

TOOLS = Tools()

def planner_node(state: AgentState):
    user_input = state["user_input"]
    planning_prompt = f"""
    You are an intelligent AI planner.
    Your job is to decide which action to take based on the user's request.
    Available actions:
    - weather
      Use for weather, temperature, humidity, rain, forecast, or climate-related questions.
    - web_search
      Use for real-time information, recent events, SAP topics, news, factual lookup, or internet-based queries.
    - direct_answer
      Use when no tool is needed and the question can be answered directly from general knowledge.
    User Input:
    {user_input}
    
    Return ONLY one word:
    weather OR web_search OR direct_answer
    """
    response = MODEL.invoke([HumanMessage(content=planning_prompt)])
    action = response.content.strip().lower()

    state["current_action"] = action
    state["next_action"] = action
    state["status"] = "planned"
    state["plan"] = [f"chosen action: {action}"]
    return state

def route(state: AgentState) -> Literal["weather_node", "web_search_node", "final_node"]:
    action = state["current_action"]
    if "weather" in action:
        return "weather_node"
    if "web_search" in action:
        return "web_search_node"
    return "final_node"

def weather_node(state: AgentState):
    result = TOOLS.weather_tool(city="Hyderabad", country="India", days=1)

    state["tool_result"] = result
    state["tools_used"].append({
        "tool": "weather_tool",
        "input": {"city": "Hyderabad", "country": "India", "days": 1},
        "output": result,
    })
    state["observations"].append({"type": "weather", "data": result})
    state["status"] = "tool_executed"
    return state

def web_search_node(state: AgentState):
    query = state["user_input"]
    result = TOOLS.web_search(query=query)

    state["tool_result"] = {"results": result}
    state["tools_used"].append({
        "tool": "web_search",
        "input": {"query": query},
        "output": result,
    })
    state["observations"].append({"type": "web_search", "data": result})
    state["status"] = "tool_executed"
    return state

def final_node(state: AgentState):
    prompt = f"""
        User request:
        {state["user_input"]}
        Tool result:
        {state.get("tool_result", {})}
        Write a helpful final answer.
        """
    response = MODEL.invoke([HumanMessage(content=prompt)])

    state["final_response"] = response.content
    state["messages"].append({"role": "assistant", "content": response.content})
    state["status"] = "done"
    return state

graph = StateGraph(AgentState)

graph.add_node("planner_node", planner_node)
graph.add_node("weather_node", weather_node)
graph.add_node("web_search_node", web_search_node)
graph.add_node("final_node", final_node)

graph.set_entry_point("planner_node")

graph.add_conditional_edges(
    "planner_node",
    route
)

graph.add_edge("weather_node", "final_node")
graph.add_edge("web_search_node", "final_node")
graph.add_edge("final_node", END)

app = graph.compile()

if __name__ == "__main__":
    messages = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break
            
            messages.append({"role": "user", "content": user_input})
            
            state: AgentState = {
                "messages": messages,
                "user_input": user_input,
                "goal": "Answer the user's request",
                "plan": [],
                "current_action": "",
                "next_action": "",
                "tool_result": {},
                "tools_used": [],
                "working_memory": {},
                "long_term_memory": {},
                "observations": [],
                "last_errnvor": "",
                "retry_count": 0,
                "confidence": 0.0,
                "status": "idle",
                "final_response": "",
            }
            
            print("\n[Thinking...]")
            result = app.invoke(state)
            
            print(f"\nAgent: {result['final_response']}")
            
            # Sync messages to include the assistant's response
            messages = result["messages"]
            
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")