#!/bin/bash
# P5 · curl demo：手动启动 server 后用 curl 发请求。
#
# 用法:
#   1. 启动 server:
#      ~/venvs/dev/bin/python -m nano_vllm.server.api --model models/Qwen2.5-1.5B-Instruct
#
#   2. 另开终端运行本脚本:
#      bash demo/p5_curl.sh

BASE="http://localhost:8000"
MODEL="models/Qwen2.5-1.5B-Instruct"

echo "=== 1. /v1/completions 非流式 ==="
curl -s "$BASE/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 16,
    \"temperature\": 0.0
  }" | python3 -m json.tool

echo ""
echo "=== 2. /v1/completions 流式 ==="
curl -sN "$BASE/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 16,
    \"temperature\": 0.0,
    \"stream\": true
  }"

echo ""
echo "=== 3. /v1/chat/completions 非流式 ==="
curl -s "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}],
    \"max_tokens\": 16,
    \"temperature\": 0.0
  }" | python3 -m json.tool

echo ""
echo "=== 4. /v1/chat/completions 流式 ==="
curl -sN "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}],
    \"max_tokens\": 16,
    \"temperature\": 0.0,
    \"stream\": true
  }"

echo ""
echo "Demo 完成。"