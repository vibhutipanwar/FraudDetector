"""
Data models and schema definitions for the FDAM system.
Defines Pydantic models for API and internal data structures.
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from uuid import uuid4

def generate_id(prefix: str) -> str:
    """Generate a unique ID with a prefix."""
    return f"{prefix}_{uuid4().hex}"

class UserIdentity(BaseModel):
    """User identity information."""
    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    country: str
    account_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "user_1234abcd",
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "country": "US",
                "account_id": "acct_5678efgh"
            }
        }

class DeviceInfo(BaseModel):
    """Device information for transactions."""
    device_id: str
    ip_address: str
    user_agent: str
    is_new: bool = False
    is_emulator: bool = False
    risk_score: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "device_id": "dev_9012ijkl",
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "is_new": False,
                "is_emulator": False,
                "risk_score": 0.2
            }
        }

class LocationInfo(BaseModel):
    """Location information for transactions."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    city: Optional[str] = None
    is_mismatch: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "country": "US",
                "city": "San Francisco",
                "is_mismatch": False
            }
        }

class TransactionData(BaseModel):
    """Transaction data model."""
    transaction_id: str = Field(default_factory=lambda: generate_id("tx"))
    payer: UserIdentity
    payee: UserIdentity
    amount: float
    currency: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    description: Optional[str] = None
    category: Optional[str] = None
    device_info: Optional[DeviceInfo] = None
    location_info: Optional[LocationInfo] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "tx_1234abcd",
                "payer": {
                    "id": "user_1234abcd",
                    "name": "Jane Doe",
                    "country": "US",
                    "account_id": "acct_5678efgh"
                },
                "payee": {
                    "id": "merchant_5678efgh",
                    "name": "Online Store",
                    "country": "US",
                    "account_id": "acct_9012ijkl"
                },
                "amount": 99.99,
                "currency": "USD",
                "timestamp": "2023-01-01T12:00:00Z",
                "description": "Online purchase",
                "category": "retail"
            }
        }

class FraudPrediction(BaseModel):
    """Fraud prediction result."""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    fraud_category: Optional[str] = None
    model_version: str
    prediction_time: str
    features_used: List[str]
    explanation: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "tx_1234abcd",
                "is_fraud": False,
                "fraud_probability": 0.05,
                "fraud_category": None,
                "model_version": "20230101120000",
                "prediction_time": "2023-01-01T12:00:05Z",
                "features_used": ["amount", "is_international", "is_new_device"]
            }
        }

class RuleResult(BaseModel):
    """Rule evaluation result."""
    rule_id: str
    rule_name: str
    is_triggered: bool
    score: float
    category: str
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "rule_id": "rule_1234abcd",
                "rule_name": "High Amount International Transfer",
                "is_triggered": True,
                "score": 0.8,
                "category": "amount_velocity",
                "metadata": {
                    "threshold": 1000,
                    "actual_value": 1500
                }
            }
        }

class FraudAnalysisResult(BaseModel):
    """Combined fraud analysis result."""
    transaction_id: str
    is_fraud: bool
    confidence: float
    category: Optional[str] = None
    model_prediction: Optional[FraudPrediction] = None
    rule_results: Optional[List[RuleResult]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "tx_1234abcd",
                "is_fraud": False,
                "confidence": 0.95,
                "category": None,
                "timestamp": "2023-01-01T12:00:10Z"
            }
        }

class FraudReport(BaseModel):
    """Fraud report submitted by user or system."""
    report_id: str = Field(default_factory=lambda: generate_id("report"))
    transaction_id: str
    reporter_id: str
    reporter_type: str = "user"  # "user" or "system"
    reason: str
    details: Optional[str] = None
    reported_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "pending"  # "pending", "investigating", "confirmed", "rejected"
    attachments: Optional[List[str]] = None
    
    @validator('reporter_type')
    def validate_reporter_type(cls, v):
        if v not in ["user", "system"]:
            raise ValueError("reporter_type must be either 'user' or 'system'")
        return v
        
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ["pending", "investigating", "confirmed", "rejected"]
        if v not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "report_id": "report_abcd1234",
                "transaction_id": "tx_1234abcd",
                "reporter_id": "user_1234abcd",
                "reporter_type": "user",
                "reason": "Unauthorized transaction",
                "details": "I did not make this purchase",
                "reported_at": "2023-01-02T15:30:00Z",
                "status": "pending"
            }
        }

class CaseManagement(BaseModel):
    """Case management for fraud investigations."""
    case_id: str = Field(default_factory=lambda: generate_id("case"))
    transaction_ids: List[str]
    user_ids: List[str]
    assigned_to: Optional[str] = None
    priority: str = "medium"  # "low", "medium", "high", "critical"
    status: str = "open"  # "open", "investigating", "pending_info", "resolved", "closed"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    resolution: Optional[str] = None
    notes: Optional[List[Dict[str, Any]]] = None
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ["low", "medium", "high", "critical"]
        if v not in valid_priorities:
            raise ValueError(f"priority must be one of {valid_priorities}")
        return v
        
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ["open", "investigating", "pending_info", "resolved", "closed"]
        if v not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "case_1234abcd",
                "transaction_ids": ["tx_1234abcd", "tx_5678efgh"],
                "user_ids": ["user_1234abcd"],
                "assigned_to": "analyst_5678efgh",
                "priority": "high",
                "status": "investigating",
                "created_at": "2023-01-02T16:00:00Z",
                "updated_at": "2023-01-03T10:15:00Z",
                "notes": [
                    {
                        "author": "analyst_5678efgh",
                        "timestamp": "2023-01-03T10:15:00Z",
                        "content": "Contacted customer for additional information"
                    }
                ]
            }
        }

class RuleDefinition(BaseModel):
    """Fraud detection rule definition."""
    rule_id: str = Field(default_factory=lambda: generate_id("rule"))
    name: str
    description: str
    category: str
    condition: Dict[str, Any]  # JSON logic format
    score: float
    is_active: bool = True
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    author: str
    version: int = 1
    
    class Config:
        schema_extra = {
            "example": {
                "rule_id": "rule_1234abcd",
                "name": "High Amount International Transfer",
                "description": "Flags high-value transactions across different countries",
                "category": "amount_velocity",
                "condition": {
                    "and": [
                        {">=": [{"var": "amount"}, 1000]},
                        {"!=": [{"var": "payer.country"}, {"var": "payee.country"}]}
                    ]
                },
                "score": 0.8,
                "is_active": True,
                "created_at": "2023-01-01T12:00:00Z",
                "updated_at": "2023-01-01T12:00:00Z",
                "author": "admin_1234abcd",
                "version": 1
            }
        }

class UserRiskProfile(BaseModel):
    """User risk profile with historical fraud data."""
    user_id: str
    risk_score: float = 0.0
    historical_transactions: int = 0
    fraud_reports_filed: int = 0
    fraud_reports_against: int = 0
    suspicious_activities: int = 0
    last_fraud_date: Optional[str] = None
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    risk_factors: List[str] = []
    tier: str = "standard"  # "low_risk", "standard", "high_risk", "blocked"
    
    @validator('tier')
    def validate_tier(cls, v):
        valid_tiers = ["low_risk", "standard", "high_risk", "blocked"]
        if v not in valid_tiers:
            raise ValueError(f"tier must be one of {valid_tiers}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_1234abcd",
                "risk_score": 0.2,
                "historical_transactions": 157,
                "fraud_reports_filed": 1,
                "fraud_reports_against": 0,
                "suspicious_activities": 3,
                "last_updated": "2023-01-03T12:00:00Z",
                "risk_factors": ["recent_account", "multiple_devices"],
                "tier": "standard"
            }
        }

class APIRequest(BaseModel):
    """Base API request model."""
    request_id: str = Field(default_factory=lambda: generate_id("req"))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    api_key_id: str
    client_info: Optional[Dict[str, Any]] = None

class FraudCheckRequest(APIRequest):
    """Request for fraud check API."""
    transaction: TransactionData
    check_type: str = "standard"  # "standard", "enhanced", "minimal"
    
    @validator('check_type')
    def validate_check_type(cls, v):
        valid_types = ["standard", "enhanced", "minimal"]
        if v not in valid_types:
            raise ValueError(f"check_type must be one of {valid_types}")
        return v

class FraudCheckResponse(BaseModel):
    """Response from fraud check API."""
    request_id: str
    transaction_id: str
    result: FraudAnalysisResult
    processing_time: float  # in milliseconds
    status: str = "success"  # "success", "error", "partial"
    error_message: Optional[str] = None

class BulkFraudCheckRequest(APIRequest):
    """Request for bulk fraud check API."""
    transactions: List[TransactionData]
    check_type: str = "standard"  # "standard", "enhanced", "minimal"
    
    @validator('check_type')
    def validate_check_type(cls, v):
        valid_types = ["standard", "enhanced", "minimal"]
        if v not in valid_types:
            raise ValueError(f"check_type must be one of {valid_types}")
        return v

class BulkFraudCheckResponse(BaseModel):
    """Response from bulk fraud check API."""
    request_id: str
    results: List[FraudCheckResponse]
    total_transactions: int
    successful_transactions: int
    failed_transactions: int
    processing_time: float  # in milliseconds
    status: str = "success"  # "success", "error", "partial"
    error_message: Optional[str] = None

class Metrics(BaseModel):
    """System performance metrics."""
    time_period: str  # "hour", "day", "week", "month"
    start_time: str
    end_time: str
    total_transactions: int
    flagged_transactions: int
    confirmed_fraud: int
    false_positives: int
    model_accuracy: float
    average_response_time: float  # in milliseconds
    
    class Config:
        schema_extra = {
            "example": {
                "time_period": "day",
                "start_time": "2023-01-01T00:00:00Z",
                "end_time": "2023-01-01T23:59:59Z",
                "total_transactions": 15782,
                "flagged_transactions": 423,
                "confirmed_fraud": 37,
                "false_positives": 386,
                "model_accuracy": 0.912,
                "average_response_time": 156.3
            }
        }
