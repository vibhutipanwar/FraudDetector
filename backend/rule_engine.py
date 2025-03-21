"""
Rule Engine for Fraud Detection System
Implements a configurable rule-based system to detect potential fraud patterns
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class RuleResult(BaseModel):
    """Result of a single rule evaluation"""
    rule_id: str
    triggered: bool
    confidence: float
    reason: str

class RuleEngineResult(BaseModel):
    """Combined result from all rules"""
    is_fraud: bool
    confidence: float
    triggered_rules: List[RuleResult]
    processing_time_ms: float

class Rule:
    """Base class for fraud detection rules"""
    def __init__(self, rule_id: str, description: str, confidence: float = 0.5):
        self.rule_id = rule_id
        self.description = description
        self.confidence = confidence
    
    def evaluate(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RuleResult:
        """Evaluate the rule against a transaction"""
        raise NotImplementedError("Rule subclasses must implement evaluate")

class VelocityRule(Rule):
    """Detects unusual transaction frequency"""
    def __init__(self, rule_id: str, description: str, 
                 time_window_minutes: int, 
                 max_transactions: int,
                 confidence: float = 0.7):
        super().__init__(rule_id, description, confidence)
        self.time_window_minutes = time_window_minutes
        self.max_transactions = max_transactions
    
    def evaluate(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RuleResult:
        redis_client = context.get('redis_client')
        if not redis_client:
            logger.error("Redis client not available for VelocityRule")
            return RuleResult(
                rule_id=self.rule_id,
                triggered=False,
                confidence=0.0,
                reason="Velocity check failed - data unavailable"
            )
        
        user_id = transaction['payer'].get('id')
        key = f"tx_count:{user_id}"
        
        # Increment the counter for this user
        count = redis_client.incr(key)
        # Set expiration if this is a new key
        if count == 1:
            redis_client.expire(key, self.time_window_minutes * 60)
        
        if count > self.max_transactions:
            return RuleResult(
                rule_id=self.rule_id,
                triggered=True,
                confidence=self.confidence,
                reason=f"User made {count} transactions in {self.time_window_minutes} minutes"
            )
        
        return RuleResult(
            rule_id=self.rule_id,
            triggered=False,
            confidence=0.0,
            reason="Transaction frequency within normal limits"
        )

class AmountRule(Rule):
    """Detects unusual transaction amounts"""
    def __init__(self, rule_id: str, description: str, 
                 threshold_amount: float,
                 confidence: float = 0.6):
        super().__init__(rule_id, description, confidence)
        self.threshold_amount = threshold_amount
    
    def evaluate(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RuleResult:
        amount = transaction.get('amount', 0)
        
        if amount > self.threshold_amount:
            return RuleResult(
                rule_id=self.rule_id,
                triggered=True,
                confidence=self.confidence,
                reason=f"Transaction amount (${amount}) exceeds threshold (${self.threshold_amount})"
            )
        
        return RuleResult(
            rule_id=self.rule_id,
            triggered=False,
            confidence=0.0,
            reason="Transaction amount within normal range"
        )

class ThresholdAvoidanceRule(Rule):
    """Detects transactions just below common thresholds"""
    def __init__(self, rule_id: str, description: str, 
                 threshold_amount: float,
                 delta: float = 50.0,
                 confidence: float = 0.5):
        super().__init__(rule_id, description, confidence)
        self.threshold_amount = threshold_amount
        self.delta = delta
    
    def evaluate(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RuleResult:
        amount = transaction.get('amount', 0)
        
        if (self.threshold_amount - self.delta) < amount < self.threshold_amount:
            return RuleResult(
                rule_id=self.rule_id,
                triggered=True,
                confidence=self.confidence,
                reason=f"Transaction amount (${amount}) just below threshold (${self.threshold_amount})"
            )
        
        return RuleResult(
            rule_id=self.rule_id,
            triggered=False,
            confidence=0.0,
            reason="Transaction amount not suspiciously close to thresholds"
        )

class GeoVelocityRule(Rule):
    """Detects impossible travel scenarios"""
    def __init__(self, rule_id: str, description: str, confidence: float = 0.8):
        super().__init__(rule_id, description, confidence)
    
    def evaluate(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RuleResult:
        redis_client = context.get('redis_client')
        if not redis_client:
            return RuleResult(
                rule_id=self.rule_id,
                triggered=False,
                confidence=0.0,
                reason="Geo velocity check failed - data unavailable"
            )
        
        user_id = transaction['payer'].get('id')
        current_ip = transaction['payer'].get('ip_address')
        current_time = datetime.fromisoformat(transaction.get('timestamp').replace('Z', '+00:00'))
        
        # Get last transaction info
        last_tx_key = f"last_tx:{user_id}"
        last_tx_data = redis_client.get(last_tx_key)
        
        if not last_tx_data:
            # First transaction, save data for future checks
            redis_client.set(last_tx_key, f"{current_ip}:{current_time.isoformat()}")
            return RuleResult(
                rule_id=self.rule_id,
                triggered=False,
                confidence=0.0,
                reason="First transaction from this user"
            )
        
        # Parse last transaction data
        last_ip, last_time_str = last_tx_data.decode().split(':')
        last_time = datetime.fromisoformat(last_time_str)
        
        # Update for next check
        redis_client.set(last_tx_key, f"{current_ip}:{current_time.isoformat()}")
        
        # Check if IPs are different but time is too close
        if last_ip != current_ip:
            time_diff = (current_time - last_time).total_seconds() / 3600  # hours
            
            # This is a very simplified check. A real implementation would use
            # geolocation to calculate actual travel time between locations.
            if time_diff < 1.0:  # Less than 1 hour between different locations
                return RuleResult(
                    rule_id=self.rule_id,
                    triggered=True,
                    confidence=self.confidence,
                    reason=f"Impossible travel: Different IP locations in {time_diff:.2f} hours"
                )
        
        return RuleResult(
            rule_id=self.rule_id,
            triggered=False,
            confidence=0.0,
            reason="No suspicious location changes detected"
        )

class BlacklistRule(Rule):
    """Checks against blacklisted entities"""
    def __init__(self, rule_id: str, description: str, confidence: float = 0.9):
        super().__init__(rule_id, description, confidence)
    
    def evaluate(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> RuleResult:
        redis_client = context.get('redis_client')
        if not redis_client:
            return RuleResult(
                rule_id=self.rule_id,
                triggered=False,
                confidence=0.0,
                reason="Blacklist check failed - data unavailable"
            )
        
        user_id = transaction['payer'].get('id')
        ip_address = transaction['payer'].get('ip_address')
        
        # Check if user is blacklisted
        if redis_client.sismember("blacklist:users", user_id):
            return RuleResult(
                rule_id=self.rule_id,
                triggered=True,
                confidence=self.confidence,
                reason=f"User {user_id} is blacklisted"
            )
        
        # Check if IP is blacklisted
        if redis_client.sismember("blacklist:ips", ip_address):
            return RuleResult(
                rule_id=self.rule_id,
                triggered=True,
                confidence=self.confidence,
                reason=f"IP address {ip_address} is blacklisted"
            )
        
        return RuleResult(
            rule_id=self.rule_id,
            triggered=False,
            confidence=0.0,
            reason="No blacklisted entities found"
        )

class RuleEngine:
    """Main rule engine that coordinates rule evaluation"""
    def __init__(self, redis_client: redis.Redis):
        self.rules: List[Rule] = []
        self.redis_client = redis_client
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize the default rule set"""
        self.rules = [
            VelocityRule(
                rule_id="VELOCITY_CHECK",
                description="Detects multiple transactions in a short time window",
                time_window_minutes=60,
                max_transactions=5
            ),
            AmountRule(
                rule_id="HIGH_AMOUNT",
                description="Detects unusually high transaction amounts",
                threshold_amount=5000.0
            ),
            ThresholdAvoidanceRule(
                rule_id="THRESHOLD_AVOIDANCE",
                description="Detects transactions just below reporting thresholds",
                threshold_amount=10000.0,
                delta=100.0
            ),
            GeoVelocityRule(
                rule_id="IMPOSSIBLE_TRAVEL",
                description="Detects transactions from different locations in short time periods"
            ),
            BlacklistRule(
                rule_id="BLACKLIST_CHECK",
                description="Checks against known fraudulent entities"
            )
        ]
    
    def add_rule(self, rule: Rule):
        """Add a custom rule to the engine"""
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID"""
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        return len(self.rules) < initial_count
    
    def evaluate(self, transaction: Dict[str, Any]) -> RuleEngineResult:
        """Evaluate all rules against a transaction"""
        import time
        start_time = time.time()
        
        context = {
            "redis_client": self.redis_client
        }
        
        results = []
        for rule in self.rules:
            try:
                result = rule.evaluate(transaction, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {str(e)}")
                results.append(RuleResult(
                    rule_id=rule.rule_id,
                    triggered=False,
                    confidence=0.0,
                    reason=f"Rule evaluation failed: {str(e)}"
                ))
        
        # Calculate the overall fraud score
        triggered_rules = [r for r in results if r.triggered]
        
        # If any rule is triggered, calculate confidence based on triggered rules
        is_fraud = len(triggered_rules) > 0
        
        if is_fraud:
            # Use the highest confidence from triggered rules
            confidence = max(r.confidence for r in triggered_rules)
        else:
            confidence = 0.0
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return RuleEngineResult(
            is_fraud=is_fraud,
            confidence=confidence,
            triggered_rules=results,
            processing_time_ms=processing_time
        )
