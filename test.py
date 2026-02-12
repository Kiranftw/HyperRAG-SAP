import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

queries =  [{'sub_query': 'Statistical summary and key metrics of sales_data.csv', 'type': 'analysis', 'strategy': 'search for summary statistics, totals, averages, growth rates'}, {'sub_query': 'Identified market trends and forecasts in market_trends.pdf', 'type': 'trend', 'strategy': 'search for trend analyses, forecasts, sector growth patterns'}, {'sub_query': 'Correlation between sales figures and identified market trends', 'type': 'comparative', 'strategy': 'search for correlation studies, cross‑analysis, regression results'}, {'sub_query': 'Actionable recommendations derived from combined sales and market data', 'type': 'procedural', 'strategy': 'search for insights, strategic recommendations, implementation steps'}, {'sub_query': 'Data quality issues and limitations in sales_data.csv and market_trends.pdf', 'type': 'constraints', 'strategy': 'search for missing values, bias, scope limitations, reliability assessments'}]
try:
    queries = [
        item.get("sub_query") for item in queries
        if isinstance(item, dict) and "sub_query" in item
    ]
    print(list(queries))
except json.JSONDecodeError as e:
    print(f"JSONDecodeError: {e}")
    queries = []
    print(queries)