# TODO: BULDING AN SINGLE AGWNT THAT PLANS AND EXECUTES OPERATIONS THEN WE GO FOR MULTIAGENT SYSTEMS
import logging
import os
from pickletools import dis
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
