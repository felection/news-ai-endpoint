# app/models/health.py
from pydantic import BaseModel
from typing import List, Dict

class HealthStatus(BaseModel):
    status: str
    endpoints: List[Dict[str, str]]