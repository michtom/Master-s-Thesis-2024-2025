#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOGFILE="$SCRIPT_DIR/retrieve_data.log"

export PYTHONPATH="$(dirname "$SCRIPT_DIR")"

echo $SCRIPT_DIR

python "$SCRIPT_DIR/coingecko_data_retrieving.py" >> "$LOGFILE" 2>&1
python "$SCRIPT_DIR/news_api_retrieving.py" >> "$LOGFILE" 2>&1
python "$SCRIPT_DIR/reddit_comments_retrieving.py" >> "$LOGFILE" 2>&1
