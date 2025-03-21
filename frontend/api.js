version: '3.8'

services:
  api:
    build: .
    container_name: fraud-detection-api
    volumes:
      - .:/app
      - ./models:/app/models
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - MONGO_URI=mongodb://mongodb:27017
      - REDIS_HOST=redis
      - ENABLE_PERFORMANCE_MONITORING=true
      - LOG_LEVEL=DEBUG
    depends_on:
      - mongodb
      - redis
    restart: unless-stopped
    networks:
      - fraud-detection-network

  mongodb:
    image: mongo:5.0
    container_name: fraud-detection-mongodb
    volumes:
      - mongodb_data:/data/db
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_DATABASE=fraud_detection
    restart: unless-stopped
    networks:
      - fraud-detection-network
    command: ["--wiredTigerCacheSizeGB", "1"]

  redis:
    image: redis:6.2-alpine
    container_name: fraud-detection-redis
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    networks:
      - fraud-detection-network
    command: ["redis-server", "--appendonly", "yes", "--maxmemory", "256mb", "--maxmemory-policy", "allkeys-lru"]

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: fraud-detection-frontend
    volumes:
      - ./frontend:/app/frontend
      - /app/frontend/node_modules
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - fraud-detection-network

  # Worker for processing batch tasks and model training
  worker:
    build: .
    container_name: fraud-detection-worker
    volumes:
      - .:/app
      - ./models:/app/models
    environment:
      - ENVIRONMENT=development
      - MONGO_URI=mongodb://mongodb:27017
      - REDIS_HOST=redis
      - WORKER_ROLE=true
    depends_on:
      - mongodb
      - redis
    restart: unless-stopped
    networks:
      - fraud-detection-network
    command: ["python", "worker.py"]

volumes:
  mongodb_data:
  redis_data:

networks:
  fraud-detection-network:
    driver: bridge
