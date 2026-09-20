#!/bin/zsh
# Wait for the hourly CONUS404 extraction, then rerun the duration-matched
# comparison end to end.
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/PointMan/bin/python
until grep -q "^done" raw/conus404_hourly_log.txt 2>/dev/null; do sleep 120; done
echo "=== hourly extraction complete, rerunning ==="
$PY analyze_conus404_seasonality.py hourly > compiled/conus404_hourly_analysis_report.txt 2>&1
$PY plot_gauge_vs_conus404.py >> compiled/conus404_hourly_analysis_report.txt 2>&1
echo "=== rerun complete ==="
