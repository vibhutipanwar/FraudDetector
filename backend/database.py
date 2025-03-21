"""
Database connection and operations module for the FDAM system.
Handles MongoDB and Redis connections and provides data access methods.
"""
import asyncio
import logging
import motor.motor_asyncio
import redis.asyncio as redis
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pymongo import ASCENDING, DESCENDING
import json

from .config import config

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """MongoDB connection manager for the FDAM system."""
    
    def __init__(self):
        """Initialize database connections."""
        self.mongo_client = None
        self.redis_client = None
        self.db = None
        self.collections = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize database connections asynchronously."""
        if self.initialized:
            return
            
        # Connect to MongoDB
        try:
            mongo_uri = config.get("database.mongodb.uri")
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                mongo_uri,
                maxPoolSize=config.get("database.mongodb.connection_pool_size")
            )
            
            db_name = config.get("database.mongodb.db_name")
            self.db = self.mongo_client[db_name]
            
            # Initialize collections
            collections_config = config.get("database.mongodb.collections")
            for name, collection in collections_config.items():
                self.collections[name] = self.db[collection]
                
            # Create indexes
            await self._create_indexes()
            logger.info(f"MongoDB connection established: {mongo_uri}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
            
        # Connect to Redis
        try:
            redis_host = config.get("database.redis.host")
            redis_port = config.get("database.redis.port")
            redis_db = config.get("database.redis.db")
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            logger.info(f"Redis connection established: {redis_host}:{redis_port}/{redis_db}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
            
        self.initialized = True
        
    async def _create_indexes(self):
        """Create necessary indexes in MongoDB collections."""
        # Transactions collection indexes
        await self.collections["transactions"].create_index([("transaction_id", ASCENDING)], unique=True)
        await self.collections["transactions"].create_index([("timestamp", DESCENDING)])
        await self.collections["transactions"].create_index([("payer.id", ASCENDING), ("timestamp", DESCENDING)])
        await self.collections["transactions"].create_index([("payee.id", ASCENDING), ("timestamp", DESCENDING)])
        await self.collections["transactions"].create_index([("fraud_analysis.is_fraud", ASCENDING), ("timestamp", DESCENDING)])
        
        # Fraud reports collection indexes
        await self.collections["fraud_reports"].create_index([("report_id", ASCENDING)], unique=True)
        await self.collections["fraud_reports"].create_index([("transaction_id", ASCENDING)])
        await self.collections["fraud_reports"].create_index([("reporter_id", ASCENDING)])
        await self.collections["fraud_reports"].create_index([("reported_at", DESCENDING)])
        
        # Rules collection indexes
        await self.collections["rules"].create_index([("rule_id", ASCENDING)], unique=True)
        await self.collections["rules"].create_index([("category", ASCENDING)])
        
        # Model metrics collection indexes
        await self.collections["model_metrics"].create_index([("timestamp", DESCENDING)])
        await self.collections["model_metrics"].create_index([("model_version", ASCENDING), ("timestamp", DESCENDING)])
        
    async def close(self):
        """Close database connections."""
        if self.mongo_client:
            self.mongo_client.close()
            
        if self.redis_client:
            await self.redis_client.close()
            
        self.initialized = False
        logger.info("Database connections closed")


class TransactionRepository:
    """Repository for transaction operations."""
    
    def __init__(self, db_conn: DatabaseConnection):
        """Initialize with database connection."""
        self.db_conn = db_conn
        self.collection = db_conn.collections["transactions"]
        self.redis = db_conn.redis_client
        
    async def insert_transaction(self, transaction: Dict[str, Any]) -> str:
        """
        Insert a new transaction into the database.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Transaction ID
        """
        # Ensure transaction has all required fields
        if "timestamp" not in transaction:
            transaction["timestamp"] = datetime.utcnow().isoformat()
            
        # Insert into MongoDB
        result = await self.collection.insert_one(transaction)
        transaction_id = transaction.get("transaction_id")
        
        # Cache transaction in Redis for quick lookup
        cache_key = f"tx:{transaction_id}"
        await self.redis.set(
            cache_key,
            json.dumps(transaction),
            ex=config.get("database.redis.cache_ttl")
        )
        
        return transaction_id
        
    async def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a transaction by ID.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Transaction data or None if not found
        """
        # Try to get from Redis cache first
        cache_key = f"tx:{transaction_id}"
        cached_data = await self.redis.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
            
        # If not in cache, get from MongoDB
        transaction = await self.collection.find_one({"transaction_id": transaction_id})
        
        if transaction:
            # Cache for future lookups
            await self.redis.set(
                cache_key,
                json.dumps(self._prepare_for_json(transaction)),
                ex=config.get("database.redis.cache_ttl")
            )
            
        return transaction
        
    async def update_transaction(self, transaction_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a transaction.
        
        Args:
            transaction_id: Transaction ID
            update_data: Data to update
            
        Returns:
            True if successful
        """
        result = await self.collection.update_one(
            {"transaction_id": transaction_id},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            # Invalidate cache
            cache_key = f"tx:{transaction_id}"
            await self.redis.delete(cache_key)
            return True
            
        return False
        
    async def get_user_transactions(
        self, 
        user_id: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        skip: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get transactions for a specific user within a time range.
        
        Args:
            user_id: User ID
            start_time: Start of time range
            end_time: End of time range
            limit: Max number of transactions
            skip: Number of transactions to skip
            
        Returns:
            List of transactions and total count
        """
        query = {"payer.id": user_id}
        
        if start_time or end_time:
            time_query = {}
            if start_time:
                time_query["$gte"] = start_time.isoformat()
            if end_time:
                time_query["$lte"] = end_time.isoformat()
            query["timestamp"] = time_query
            
        # Get count first
        total_count = await self.collection.count_documents(query)
        
        # Get transactions
        cursor = self.collection.find(query)
        cursor.sort("timestamp", DESCENDING)
        cursor.skip(skip).limit(limit)
        
        transactions = await cursor.to_list(length=limit)
        return transactions, total_count
        
    async def get_historical_data(self, user_id: str) -> Dict[str, Any]:
        """
        Get historical transaction data for a user for feature engineering.
        
        Args:
            user_id: User ID
            
        Returns:
            Historical data summary
        """
        # Try to get from Redis cache first
        cache_key = f"hist:{user_id}"
        cached_data = await self.redis.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
            
        # Calculate time ranges
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        
        # Count recent transactions
        tx_count_1h = await self.collection.count_documents({
            "payer.id": user_id,
            "timestamp": {"$gte": one_hour_ago.isoformat()}
        })
        
        tx_count_24h = await self.collection.count_documents({
            "payer.id": user_id,
            "timestamp": {"$gte": one_day_ago.isoformat()}
        })
        
        # Get average transaction amount
        pipeline = [
            {"$match": {"payer.id": user_id}},
            {"$group": {
                "_id": None,
                "avg_amount": {"$avg": "$amount"},
                "transaction_count": {"$sum": 1},
                "first_transaction": {"$min": "$timestamp"}
            }}
        ]
        
        result = await self.collection.aggregate(pipeline).to_list(length=1)
        
        if result:
            avg_amount = result[0].get("avg_amount", 0)
            transaction_count = result[0].get("transaction_count", 0)
            first_transaction = result[0].get("first_transaction", now.isoformat())
        else:
            avg_amount = 0
            transaction_count = 0
            first_transaction = now.isoformat()
            
        # Compile historical data
        historical_data = {
            "tx_count_1h": tx_count_1h,
            "tx_count_24h": tx_count_24h,
            "avg_amount": avg_amount,
            "transaction_count": transaction_count,
            "first_transaction": first_transaction,
            "account_age_days": (now - datetime.fromisoformat(first_transaction.replace('Z', '+00:00'))).days
        }
        
        # Cache for future lookups
        await self.redis.set(
            cache_key,
            json.dumps(historical_data),
            ex=config.get("database.redis.cache_ttl")
        )
        
        return historical_data
    
    async def get_fraud_statistics(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get fraud statistics for a time period.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Fraud statistics
        """
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=30)
            
        # Create time range query
        time_query = {
            "$gte": start_time.isoformat(),
            "$lte": end_time.isoformat()
        }
        
        # Try to get from Redis cache first
        cache_key = f"fraud_stats:{start_time.isoformat()}:{end_time.isoformat()}"
        cached_data = await self.redis.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
            
        # Get transaction statistics from MongoDB
        pipeline = [
            {"$match": {"timestamp": time_query}},
            {"$group": {
                "_id": None,
                "total_transactions": {"$sum": 1},
                "total_amount": {"$sum": "$amount"},
                "avg_amount": {"$avg": "$amount"}
            }}
        ]
        
        tx_stats = await self.collection.aggregate(pipeline).to_list(length=1)
        tx_stats = tx_stats[0] if tx_stats else {
            "total_transactions": 0,
            "total_amount": 0,
            "avg_amount": 0
        }
        
        # Get fraud statistics
        fraud_pipeline = [
            {"$match": {
                "timestamp": time_query,
                "fraud_analysis.is_fraud": True
            }},
            {"$group": {
                "_id": None,
                "fraud_transactions": {"$sum": 1},
                "fraud_amount": {"$sum": "$amount"},
                "avg_fraud_amount": {"$avg": "$amount"}
            }}
        ]
        
        fraud_stats = await self.collection.aggregate(fraud_pipeline).to_list(length=1)
        fraud_stats = fraud_stats[0] if fraud_stats else {
            "fraud_transactions": 0,
            "fraud_amount": 0,
            "avg_fraud_amount": 0
        }
        
        # Get fraud by category
        category_pipeline = [
            {"$match": {
                "timestamp": time_query,
                "fraud_analysis.is_fraud": True
            }},
            {"$group": {
                "_id": "$fraud_analysis.category",
                "count": {"$sum": 1},
                "amount": {"$sum": "$amount"}
            }},
            {"$sort": {"count": -1}}
        ]
        
        fraud_by_category = await self.collection.aggregate(category_pipeline).to_list(length=100)
        fraud_by_category = [
            {"category": item["_id"], "count": item["count"], "amount": item["amount"]}
            for item in fraud_by_category
        ]
        
        # Compile statistics
        statistics = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "transactions": {
                "total_count": tx_stats.get("total_transactions", 0),
                "total_amount": tx_stats.get("total_amount", 0),
                "avg_amount": tx_stats.get("avg_amount", 0)
            },
            "fraud": {
                "fraud_count": fraud_stats.get("fraud_transactions", 0),
                "fraud_amount": fraud_stats.get("fraud_amount", 0),
                "avg_fraud_amount": fraud_stats.get("avg_fraud_amount", 0),
                "fraud_rate": (fraud_stats.get("fraud_transactions", 0) / tx_stats.get("total_transactions", 1)) * 100 if tx_stats.get("total_transactions", 0) > 0 else 0,
                "fraud_amount_rate": (fraud_stats.get("fraud_amount", 0) / tx_stats.get("total_amount", 1)) * 100 if tx_stats.get("total_amount", 0) > 0 else 0
            },
            "fraud_by_category": fraud_by_category
        }
        
        # Cache for future lookups (short TTL as these are frequently changing stats)
        await self.redis.set(
            cache_key,
            json.dumps(statistics),
            ex=300  # 5 minutes TTL
        )
        
        return statistics

    def _prepare_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MongoDB data to JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items() if k != '_id'}
        elif isinstance(data, list):
            return [self._prepare_for_json(i) for i in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data


class FraudReportRepository:
    """Repository for fraud report operations."""
    
    def __init__(self, db_conn: DatabaseConnection):
        """Initialize with database connection."""
        self.db_conn = db_conn
        self.collection = db_conn.collections["fraud_reports"]
        self.redis = db_conn.redis_client
        
    async def create_report(self, report_data: Dict[str, Any]) -> str:
        """
        Create a new fraud report.
        
        Args:
            report_data: Report data
            
        Returns:
            Report ID
        """
        # Ensure report has timestamp
        if "reported_at" not in report_data:
            report_data["reported_at"] = datetime.utcnow().isoformat()
            
        # Insert into MongoDB
        result = await self.collection.insert_one(report_data)
        report_id = report_data.get("report_id")
        
        # Invalidate related caches
        transaction_id = report_data.get("transaction_id")
        if transaction_id:
            await self.redis.delete(f"tx:{transaction_id}")
            
        return report_id
        
    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a report by ID.
        
        Args:
            report_id: Report ID
            
        Returns:
            Report data or None if not found
        """
        return await self.collection.find_one({"report_id": report_id})
        
    async def update_report_status(self, report_id: str, status: str, resolution_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a report's status.
        
        Args:
            report_id: Report ID
            status: New status
            resolution_details: Optional resolution details
            
        Returns:
            True if successful
        """
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if resolution_details:
            update_data["resolution"] = resolution_details
            
        result = await self.collection.update_one(
            {"report_id": report_id},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
        
    async def get_reports_for_transaction(self, transaction_id: str) -> List[Dict[str, Any]]:
        """
        Get all reports for a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            List of reports
        """
        cursor = self.collection.find({"transaction_id": transaction_id})
        cursor.sort("reported_at", DESCENDING)
        return await cursor.to_list(length=100)
        
    async def get_pending_reports(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get pending reports that need processing.
        
        Args:
            limit: Maximum number of reports to return
            
        Returns:
            List of pending reports
        """
        cursor = self.collection.find({"status": "pending"})
        cursor.sort("reported_at", ASCENDING)
        cursor.limit(limit)
        return await cursor.to_list(length=limit)


class RuleRepository:
    """Repository for fraud detection rules."""
    
    def __init__(self, db_conn: DatabaseConnection):
        """Initialize with database connection."""
        self.db_conn = db_conn
        self.collection = db_conn.collections["rules"]
        self.redis = db_conn.redis_client
        
    async def get_active_rules(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get active fraud detection rules.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of active rules
        """
        # Try to get from Redis cache first
        cache_key = f"rules:{category if category else 'all'}"
        cached_data = await self.redis.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
            
        # Query MongoDB
        query = {"active": True}
        if category:
            query["category"] = category
            
        cursor = self.collection.find(query)
        rules = await cursor.to_list(length=1000)
        
        # Cache for future lookups
        await self.redis.set(
            cache_key,
            json.dumps(rules),
            ex=config.get("database.redis.rules_cache_ttl", 300)
        )
        
        return rules
        
    async def create_rule(self, rule_data: Dict[str, Any]) -> str:
        """
        Create a new rule.
        
        Args:
            rule_data: Rule data
            
        Returns:
            Rule ID
        """
        # Set defaults if not provided
        if "created_at" not in rule_data:
            rule_data["created_at"] = datetime.utcnow().isoformat()
            
        if "active" not in rule_data:
            rule_data["active"] = True
            
        # Insert into MongoDB
        result = await self.collection.insert_one(rule_data)
        rule_id = rule_data.get("rule_id")
        
        # Invalidate cache
        category = rule_data.get("category")
        await self.redis.delete(f"rules:{category if category else 'all'}")
        await self.redis.delete("rules:all")
        
        return rule_id
        
    async def update_rule(self, rule_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a rule.
        
        Args:
            rule_id: Rule ID
            update_data: Data to update
            
        Returns:
            True if successful
        """
        # Get current rule to identify category for cache invalidation
        current_rule = await self.collection.find_one({"rule_id": rule_id})
        
        result = await self.collection.update_one(
            {"rule_id": rule_id},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            # Invalidate cache
            if current_rule:
                category = current_rule.get("category")
                await self.redis.delete(f"rules:{category if category else 'all'}")
            await self.redis.delete("rules:all")
            return True
            
        return False
        
    async def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a rule.
        
        Args:
            rule_id: Rule ID
            
        Returns:
            True if successful
        """
        # Get current rule to identify category for cache invalidation
        current_rule = await self.collection.find_one({"rule_id": rule_id})
        
        result = await self.collection.delete_one({"rule_id": rule_id})
        
        if result.deleted_count > 0:
            # Invalidate cache
            if current_rule:
                category = current_rule.get("category")
                await self.redis.delete(f"rules:{category if category else 'all'}")
            await self.redis.delete("rules:all")
            return True
            
        return False


class ModelMetricsRepository:
    """Repository for model performance metrics."""
    
    def __init__(self, db_conn: DatabaseConnection):
        """Initialize with database connection."""
        self.db_conn = db_conn
        self.collection = db_conn.collections["model_metrics"]
        
    async def record_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        Record model performance metrics.
        
        Args:
            metrics: Metrics data
            
        Returns:
            Record ID
        """
        # Ensure metrics have timestamp
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.utcnow().isoformat()
            
        # Insert into MongoDB
        result = await self.collection.insert_one(metrics)
        return str(result.inserted_id)
        
    async def get_metrics_history(
        self,
        model_version: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get model metrics history.
        
        Args:
            model_version: Optional model version filter
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of records
            
        Returns:
            List of metrics records
        """
        query = {}
        
        if model_version:
            query["model_version"] = model_version
            
        if start_time or end_time:
            time_query = {}
            if start_time:
                time_query["$gte"] = start_time.isoformat()
            if end_time:
                time_query["$lte"] = end_time.isoformat()
            query["timestamp"] = time_query
            
        cursor = self.collection.find(query)
        cursor.sort("timestamp", DESCENDING)
        cursor.limit(limit)
        
        return await cursor.to_list(length=limit)
        
    async def get_latest_metrics(self, model_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get latest metrics for a model version.
        
        Args:
            model_version: Optional model version
            
        Returns:
            Latest metrics or None if not found
        """
        query = {}
        if model_version:
            query["model_version"] = model_version
            
        cursor = self.collection.find(query)
        cursor.sort("timestamp", DESCENDING)
        cursor.limit(1)
        
        results = await cursor.to_list(length=1)
        return results[0] if results else None
        
    async def aggregate_metrics(
        self,
        group_by: str,
        metrics: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Aggregate metrics by a dimension.
        
        Args:
            group_by: Field to group by
            metrics: List of metrics to aggregate
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Aggregated metrics
        """
        match_stage = {}
        
        if start_time or end_time:
            time_query = {}
            if start_time:
                time_query["$gte"] = start_time.isoformat()
            if end_time:
                time_query["$lte"] = end_time.isoformat()
            match_stage["timestamp"] = time_query
            
        # Build aggregation pipeline
        group_stage = {
            "_id": f"${group_by}",
            "count": {"$sum": 1}
        }
        
        # Add requested metrics
        for metric in metrics:
            group_stage[f"avg_{metric}"] = {"$avg": f"${metric}"}
            group_stage[f"min_{metric}"] = {"$min": f"${metric}"}
            group_stage[f"max_{metric}"] = {"$max": f"${metric}"}
            
        pipeline = [
            {"$match": match_stage},
            {"$group": group_stage},
            {"$sort": {"count": -1}}
        ]
        
        # Execute aggregation
        results = await self.collection.aggregate(pipeline).to_list(length=100)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_result = {
                group_by: result["_id"],
                "count": result["count"]
            }
            
            for metric in metrics:
                formatted_result[metric] = {
                    "avg": result.get(f"avg_{metric}"),
                    "min": result.get(f"min_{metric}"),
                    "max": result.get(f"max_{metric}")
                }
                
            formatted_results.append(formatted_result)
            
        return formatted_results


# Create a singleton instance
db_connection = DatabaseConnection()


async def initialize_database():
    """Initialize the database connection."""
    await db_connection.initialize()
    
    
async def close_database():
    """Close the database connection."""
    await db_connection.close()


def get_transaction_repository() -> TransactionRepository:
    """Get the transaction repository instance."""
    return TransactionRepository(db_connection)
    
    
def get_fraud_report_repository() -> FraudReportRepository:
    """Get the fraud report repository instance."""
    return FraudReportRepository(db_connection)
    
    
def get_rule_repository() -> RuleRepository:
    """Get the rule repository instance."""
    return RuleRepository(db_connection)
    
    
def get_model_metrics_repository() -> ModelMetricsRepository:
    """Get the model metrics repository instance."""
    return ModelMetricsRepository(db_connection)
