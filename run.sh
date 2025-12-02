#!/bin/bash

echo "===================================="
echo "Starting Prediction Scheduler"
echo "===================================="
echo "Press Ctrl+C to stop"
echo ""

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting predictions..."
    
    # 병렬 실행
    python main.py --object cpu &
    python main.py --object memory &
    python main.py --object disk &
    
    # 모든 백그라운드 작업이 끝날 때까지 대기
    wait
    
    echo "All predictions completed. Waiting 60 seconds..."
    echo ""
    
    # 1분 대기
    sleep 60
done