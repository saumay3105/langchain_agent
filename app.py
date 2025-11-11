from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from agent import agent_executor, ChangeRequest, RiskAssessment, RiskLevel
from typing import List
from langchain_core.messages import SystemMessage
import re

app = FastAPI(
    title="Change Risk Management Agent API (Powered by Gemini)",
    description="AI-powered change risk analysis using Google Gemini",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System prompt for the agent
SYSTEM_PROMPT = """You are an expert Change Risk Management Analyst AI with deep expertise in IT operations and risk assessment.

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

@app.post("/api/analyze-change", response_model=RiskAssessment)
async def analyze_change_risk(change: ChangeRequest) -> RiskAssessment:
    """Analyze change request using Gemini AI agent"""
    try:
        query = f"""Analyze this IT change request comprehensively:

CHANGE DETAILS:
- Change ID: {change.change_id}
- Description: {change.description}
- Affected Systems: {', '.join(change.affected_systems)}
- Implementation Date: {change.implementation_date}

CHANGE ATTRIBUTES:
- Teams Involved: {change.teams_involved}
- Testing Status: {'✓ Completed' if change.testing_completed else '✗ Not Completed'}
- Rollback Plan: {'✓ Available' if change.has_rollback_plan else '✗ Missing'}
- Service Outage: {'Required ({} min)'.format(change.outage_duration_minutes) if change.service_outage_required else 'Not Required'}

Provide a thorough risk assessment."""
        
        result = await asyncio.to_thread(
            agent_executor.invoke,
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    ("user", query)
                ]
            }
        )
        
        # Extract the last message content
        output = result["messages"][-1].content
        assessment = parse_gemini_output(output, change)
        return assessment
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def parse_gemini_output(output: str, change: ChangeRequest) -> RiskAssessment:
    """Parse Gemini agent output into structured RiskAssessment"""
    output_lower = output.lower()
    
    # Extract risk score
    risk_score = extract_risk_score(output_lower)
    
    # Determine risk level
    if risk_score < 25:
        risk_level = RiskLevel.LOW
    elif risk_score < 50:
        risk_level = RiskLevel.MODERATE
    elif risk_score < 75:
        risk_level = RiskLevel.HIGH
    else:
        risk_level = RiskLevel.CRITICAL
    
    risk_factors = {
        "complexity": "moderate" if change.teams_involved > 2 else "low",
        "testing_coverage": "adequate" if change.testing_completed else "insufficient",
        "historical_pattern": "favorable",
        "service_impact": "high" if change.service_outage_required else "low"
    }
    
    recommendations = extract_recommendations(output_lower, change)
    
    return RiskAssessment(
        change_id=change.change_id,
        risk_level=risk_level,
        risk_score=risk_score,
        risk_factors=risk_factors,
        recommendations=recommendations,
        approval_required=risk_score > 50
    )

def extract_risk_score(output: str) -> int:
    """Extract numerical risk score from agent output"""
    score_match = re.search(r'(\d+)/100|score:\s*(\d+)', output)
    if score_match:
        return int(score_match.group(1) or score_match.group(2))
    return 50

def extract_recommendations(output: str, change: ChangeRequest) -> List[str]:
    """Extract actionable recommendations"""
    recommendations = []
    
    if not change.testing_completed:
        recommendations.append("Complete comprehensive testing before implementation")
    if not change.has_rollback_plan:
        recommendations.append("Develop and validate rollback procedure")
    if change.service_outage_required:
        recommendations.append("Schedule during off-peak hours (2-6 AM IST)")
    if change.teams_involved > 3:
        recommendations.append("Conduct pre-implementation coordination meeting")
    
    recommendations.append("Monitor key metrics for 24 hours post-deployment")
    return recommendations

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "gemini-2.0-flash-exp",
        "framework": "LangGraph + FastAPI"
    }
