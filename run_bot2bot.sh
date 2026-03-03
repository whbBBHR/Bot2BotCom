#!/bin/bash
# Wrapper script to run Bot2Bot with correct environment

cd "$(dirname "$0")"
source Botvenv/bin/activate
python API_2_Api_com.py
