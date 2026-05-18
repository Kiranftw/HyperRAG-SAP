import logging
import os
from typing import Dict, List, Optional
import pandas as pd
from langchain.agents import create_agent
from pydantic import BaseModel

# root path
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger()

class OrgRow(BaseModel):
    company: str
    company_code: str
    plant: Optional[str] = None
    storage_location: Optional[str] = None
    sales_organization: Optional[str] = None
    distribution_channel: Optional[str] = None
    division: Optional[str] = None
    sales_office: Optional[str] = None
    sales_group: Optional[str] = None
    warehouse: Optional[str] = None
    warehouse_number: Optional[str] = None
    assignment_type: Optional[str] = None  # important: relation/connection type
    parent_code: Optional[str] = None  # important: dependency link

class PlanStep(BaseModel):
    action: str
    entity_type: str
    parent_key: Optional[str] = None
    payload: Dict

class OrgPlan(BaseModel):
    rows: List[OrgRow]
    steps: List[PlanStep]

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
    source: Dict

# pip command to install langchain_nvidia_ai_endpoints
class Tools:
    def __init__(self):
        self.SEARCH_ENGINE = TavilySearch(
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            max_results=1,
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

    Your goal is to create a step-by-step plan to address the user's request. Consider:
    1. The user's request
    2. Available tools
    3. Data dependencies
    4. Potential constraints

    Generate a clear, actionable plan.

    User Request: {user_input}
    """

    response = MODEL.invoke([HumanMessage(content=planning_prompt)])
    plan_str = response.content.strip()
    state["plan"] = plan_str.split("\n")
    return state

# TODO: implement function calling to create the org plan from the plan steps.

def function_calling_node(state: AgentState):
    user_input = state["user_input"]
    model = ChatNVIDIA(
        model="moonshotai/kimi-k2-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=1,
        top_p=0.9,
        max_completion_tokens=16384,
        tools=[TOOLS.weather_tool, TOOLS.web_search]
    )
    messages = state["messages"]
    response = model.invoke([HumanMessage(content=user_input)])
    tool_calls =
from langchain.agents import create_agent
from pydantic import BaseModel

# root path
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger()

class OrgRow(BaseModel):
    company: str
    company_code: str
    plant: Optional[str] = None
    storage_location: Optional[str] = None
    sales_organization: Optional[str] = None
    distribution_channel: Optional[str] = None
    division: Optional[str] = None
    sales_office: Optional[str] = None
    sales_group: Optional[str] = None
    warehouse: Optional[str] = None
    warehouse_number: Optional[str] = None
    assignment_type: Optional[str] = None  # important: relation/connection type
    parent_code: Optional[str] = None  # important: dependency link

class PlanStep(BaseModel):
    action: str
    entity_type: str
    parent_key: Optional[str] = None
    payload: Dict

class OrgPlan(BaseModel):
    rows: List[OrgRow]
    steps: List[PlanStep]

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
    source: Dict
    document_ids: List[str]
    ingestion_job_id: str