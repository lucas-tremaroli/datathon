#!/bin/bash
# Test drift detection: send normal data, then drifted data, check results.

BASE_URL="${BASE_URL:-http://localhost:8000}"

NORMAL='{"students":[
  {"ieg":8.5,"iaa":8.0,"ips":6.0,"ida":6.5,"ian":7.0,"ipv":7.8,"inde":7.4,"stone":3,"age":12},
  {"ieg":9.0,"iaa":9.5,"ips":7.5,"ida":7.8,"ian":10.0,"ipv":8.4,"inde":8.0,"stone":3,"age":14},
  {"ieg":7.5,"iaa":7.0,"ips":5.0,"ida":5.5,"ian":5.0,"ipv":7.0,"inde":6.5,"stone":2,"age":10}
]}'

DRIFTED='{"students":[
  {"ieg":1.0,"iaa":1.0,"ips":1.0,"ida":1.0,"ian":1.0,"ipv":2.0,"inde":2.0,"stone":1,"age":35},
  {"ieg":2.0,"iaa":1.5,"ips":1.5,"ida":1.5,"ian":1.0,"ipv":2.5,"inde":2.5,"stone":1,"age":30},
  {"ieg":1.5,"iaa":2.0,"ips":2.0,"ida":2.0,"ian":2.0,"ipv":3.0,"inde":3.0,"stone":1,"age":40}
]}'

echo "Sending 15 normal batches (45 samples)"
for i in $(seq 1 15); do
  curl -s -X POST "$BASE_URL/api/predict" -H "Content-Type: application/json" -d "$NORMAL" > /dev/null
done

echo "Drift check (expect: no_drift)"
curl -s "$BASE_URL/api/model/drift" | python3 -m json.tool

echo "Sending 20 drifted batches (60 samples)"
for i in $(seq 1 20); do
  curl -s -X POST "$BASE_URL/api/predict" -H "Content-Type: application/json" -d "$DRIFTED" > /dev/null
done

echo "Drift check (expect: warning/alert)"
curl -s "$BASE_URL/api/model/drift" | python3 -m json.tool
