from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)



class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ChangeRequest(BaseModel):
    change_id: str
    description: str
    affected_systems: List[str]
    implementation_date: str
    teams_involved: int
    has_rollback_plan: bool
    testing_completed: bool
    service_outage_required: bool
    outage_duration_minutes: Optional[int] = 0


class RiskAssessment(BaseModel):
    change_id: str
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    risk_factors: Dict[str, str]
    recommendations: List[str]
    approval_required: bool


@tool
def query_historical_changes(query: str) -> str:
    """Query historical change records to find similar past changes and their success rates.
    Returns data about similar changes, their outcomes, and patterns."""
    return """Found 8 similar changes in the last 6 months:
    - 6 successful (75% success rate)
    - 2 failed due to insufficient testing
    - Average complexity score: 3.2/5
    - Most common risk: database schema changes"""


@tool
def check_configuration_items(systems: str) -> str:
    """Check Configuration Management Database for affected systems and their dependencies.
    Returns information about dependent services, criticality, and downstream impacts.
    """
    return f"""Systems: {systems}
    Dependencies found:
    - 4 downstream services will be affected
    - 2 are production-critical systems
    - Estimated blast radius: 250 users
    - Recovery Time Objective (RTO): 2 hours"""


@tool
def detect_conflicts(implementation_date: str) -> str:
    """Detect scheduling conflicts with other planned changes or maintenance windows.
    Returns information about overlapping changes and blackout periods."""
    return f"""Date analysis for {implementation_date}:
    - No conflicting changes scheduled
    - Outside of monthly blackout period
    - Maintenance window available: 2 AM - 6 AM IST
    - Business impact window: LOW"""


@tool
def calculate_risk_score(change_data: str) -> str:
    """Calculate numerical risk score based on multiple weighted factors.
    Returns detailed scoring breakdown and justification."""
    return """Risk Score Calculation:
    - Complexity: 15/25 (moderate)
    - Testing: 20/25 (adequate)
    - Historical: 15/20 (good track record)
    - Impact: 18/25 (significant but manageable)
    - Rollback: 8/10 (plan exists)
    Total Score: 76/100 (HIGH RISK)"""



tools = [
    query_historical_changes,
    check_configuration_items,
    detect_conflicts,
    calculate_risk_score,
]


system_prompt = """You are an expert Change Risk Management Analyst AI with deep expertise in IT operations and risk assessment.

Your task is to analyze change requests and provide comprehensive risk assessments by:
1. Querying historical data for similar changes
2. Checking configuration dependencies and impact scope
3. Detecting scheduling conflicts
4. Calculating weighted risk scores
5. Providing actionable recommendations

RISK ASSESSMENT CRITERIA:
- Complexity: Team count, technical difficulty, scope
- Historical Success: Past performance of similar changes
- Testing Coverage: Quality and extent of validation
- Service Impact: Downtime, user impact, criticality
- Mitigation: Rollback plans and safety measures

RISK LEVELS:
- LOW (0-25): Routine change, minimal impact
- MODERATE (26-50): Standard change with some risk
- HIGH (51-75): Complex change requiring approval
- CRITICAL (76-100): High-risk change requiring CAB review

Use the available tools to gather information and provide comprehensive analysis."""

agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
)
