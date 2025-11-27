import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
# from google.analytics.data_v1beta import BetaAnalyticsDataClient
# from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Metric, Dimension

def fetch_ga4_data(property_id, start_date, end_date):
    """
    Fetches data from GA4 API.
    NOTE: This is a placeholder structure. Actual implementation requires valid credentials.
    For now, we will primarily use the dummy data generator for the 'Scroll LP' scenario.
    """
    # client = BetaAnalyticsDataClient()
    # request = RunReportRequest(
    #     property=f"properties/{property_id}",
    #     date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
    #     dimensions=[Dimension(name="date"), Dimension(name="eventName")],
    #     metrics=[Metric(name="eventCount")],
    # )
    # response = client.run_report(request=request)
    # ... parsing logic ...
    
    # Returning None to trigger dummy data usage in main app for now
    return None

def generate_scroll_lp_dummy_data(days=30):
    """
    Generates dummy data for a Scroll LP scenario.
    Events: session_start, scroll (10%, 25%, 50%, 75%, 90%), click_cta, conversion
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    data = []
    
    # LP Variations for A/B testing
    lp_variants = ['/lp-scroll-v1', '/lp-scroll-v2']
    
    # Devices
    devices = ['mobile', 'desktop', 'tablet']
    
    # Traffic Sources
    sources = [
        ('google', 'cpc'), ('google', 'organic'), 
        ('facebook', 'paid_social'), ('instagram', 'paid_social'),
        ('direct', 'none'), ('email', 'newsletter')
    ]
    
    for _ in range(2000): # Generate 2000 sessions
        session_id = f"sess_{np.random.randint(100000, 999999)}"
        event_date = pd.Timestamp(np.random.choice(date_range))
        lp = np.random.choice(lp_variants)
        device = np.random.choice(devices, p=[0.7, 0.25, 0.05])
        source, medium = sources[np.random.choice(len(sources), p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])]
        
        # Derived dimensions
        ga_session_number = 1 if np.random.random() < 0.7 else np.random.randint(2, 10)
        source_medium = f"{source} / {medium}"
        
        if medium == 'cpc': channel = 'Paid Search'
        elif medium == 'organic': channel = 'Organic Search'
        elif medium == 'paid_social': channel = 'Paid Social'
        elif medium == 'newsletter': channel = 'Email'
        else: channel = 'Direct'
        
        load_time_ms = np.random.randint(500, 3000)

        # Base session event
        data.append({
            'event_date': event_date,
            'event_name': 'session_start',
            'session_id': session_id,
            'page_location': lp,
            'device_type': device,
            'utm_source': source,
            'utm_medium': medium,
            'source_medium': source_medium,
            'channel': channel,
            'ga_session_number': ga_session_number,
            'scroll_depth': 0,
            'stay_ms': 0,
            'load_time_ms': load_time_ms
        })
        
        # Simulate Scroll Depth
        # Probabilities of reaching depth: 10%, 25%, 50%, 75%, 90%
        # Mobile tends to scroll more but bounce faster? Let's keep it simple.
        max_depth_probs = [0.1, 0.15, 0.2, 0.2, 0.2, 0.15] # 0, 10, 25, 50, 75, 90
        depths = [0, 10, 25, 50, 75, 90]
        
        reached_depth_idx = np.random.choice(range(len(depths)), p=max_depth_probs)
        reached_depth = depths[reached_depth_idx]
        
        # Generate scroll events up to reached_depth
        for i in range(1, reached_depth_idx + 1):
            d = depths[i]
            data.append({
                'event_date': event_date + timedelta(seconds=np.random.randint(5, 60)),
                'event_name': f'scroll_{d}',
                'session_id': session_id,
                'page_location': lp,
                'device_type': device,
                'utm_source': source,
                'utm_medium': medium,
                'source_medium': source_medium,
                'channel': channel,
                'ga_session_number': ga_session_number,
                'scroll_depth': d,
                'stay_ms': np.random.randint(1000, 30000),
                'load_time_ms': load_time_ms
            })
            
        # Conversion Logic
        # If reached 90% or 75%, higher chance of CTA click
        if reached_depth >= 75:
            if np.random.random() < 0.15: # 15% CTR on CTA
                data.append({
                    'event_date': event_date + timedelta(seconds=np.random.randint(60, 120)),
                    'event_name': 'click_cta',
                    'session_id': session_id,
                    'page_location': lp,
                    'device_type': device,
                    'utm_source': source,
                    'utm_medium': medium,
                    'source_medium': source_medium,
                    'channel': channel,
                    'ga_session_number': ga_session_number,
                    'scroll_depth': reached_depth,
                    'stay_ms': np.random.randint(30000, 60000),
                    'load_time_ms': load_time_ms
                })
                
                if np.random.random() < 0.3: # 30% CVR after click
                    data.append({
                        'event_date': event_date + timedelta(seconds=np.random.randint(120, 300)),
                        'event_name': 'conversion',
                        'session_id': session_id,
                        'page_location': lp,
                        'device_type': device,
                        'utm_source': source,
                        'utm_medium': medium,
                        'source_medium': source_medium,
                        'channel': channel,
                        'ga_session_number': ga_session_number,
                        'scroll_depth': reached_depth,
                        'stay_ms': np.random.randint(60000, 120000),
                        'cv_type': 'purchase',
                        'load_time_ms': load_time_ms
                    })

    df = pd.DataFrame(data)
    return df
