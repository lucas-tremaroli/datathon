#!/bin/bash

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "=== Health Check ==="
curl -s "$BASE_URL/" | jq

echo -e "\n=== Model Info ==="
curl -s "$BASE_URL/api/model/info" | jq

echo -e "\n=== Single Prediction ==="
curl -s -X POST "$BASE_URL/api/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "ieg": 7.5,
    "iaa": 8.0,
    "ips": 6.5,
    "ida": 7.2,
    "ian": 8.5,
    "ipv": 6.0,
    "inde": 7.3,
    "stone": 3,
    "age": 14
  }' | jq

echo -e "\n=== Batch Prediction ==="
curl -s -X POST "$BASE_URL/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {
        "ieg": 7.5,
        "iaa": 8.0,
        "ips": 6.5,
        "ida": 7.2,
        "ian": 8.5,
        "ipv": 6.0,
        "inde": 7.3,
        "stone": 3,
        "age": 14
      },
      {
        "ieg": 5.0,
        "iaa": 4.5,
        "ips": 5.2,
        "ida": 4.8,
        "ian": 5.0,
        "ipv": 3.5,
        "inde": 4.7,
        "stone": 1,
        "age": 12
      }
    ]
  }' | jq
