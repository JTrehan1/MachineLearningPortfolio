"""Mission event timeline alignment 
You have two lists of timestamped events: one from a sensor log and one from an operator action log. 
Write a function that merges them into a single sorted timeline, and for each event annotates whether it occurred within 
5 seconds of an event in the other log."""

def timeline_alignment(sensor_log, operator_log, proximity_threshold=5):
    """
    Function to merge data from two logs and annotate events that occurred within 5 seconds of each other.
    
    :param sensor_log: List of tuples (timestamp, event) from the sensor log
    :param operator_log: List of tuples (timestamp, event) from the operator log
    :proximity_threshold: Time threshold in seconds to consider events as correlated
    
    Each log is a list of dicts: {'timestamp': float, 'event': str}
    Returns: single sorted merged log with 'source' and 'correlated' fields added.
    """
    # Tagged logs 
    tagged_sensor = [{'timestamp': event['timestamp'],
                     'event_source': 'sensor',
                     'correlated': False} for event in sensor_log
                     ]
    
    tagged_operator = [{'timestamp': event['timestamp'],
                     'event_source': 'operator',
                     'correlated': False} for event in sensor_log
                     ]
                     
    # Merge and sort the two lists 
    merged_log = tagged_sensor + tagged_operator
    sorted_log = sorted(merged_log, key=lambda x: x['timestamp'])
    
    left = 0
    n = len(sorted_log)
    
    for right in range(n):
        while sorted_log[right]['timestamp']- sorted_log[left]['timestamp'] > proximity_threshold:
            left += 1
            
        for k in range(left, right): # Search window is only within the proximity threshold
            if sorted_log[right]['event_source'] != sorted_log[left]['event_source']:
                sorted_log[right]['correlated'] = True
                sorted_log[left]['correlated'] = True
    
    return sorted_log
 
"""Alert deduplication You receive a stream of alerts. Each alert has an id, a source, a type, and a timestamp. 
    Write a function that deduplicates alerts: two alerts are duplicates if they have the same type and source, and their timestamps are within 60 seconds of each other. 
    Return only the first occurrence of each duplicate group.
"""
    
def deduplicate_alerts(alerts, threshold=60):
    """
    Function to deduplicate alerts from a stream. Each alert has an id, source, type and timestamp.
            
    Alerts are duplicates if they have the same source, type and within 60s of each other.
    Args:
        alerts (list): list of dictionary entires representing alerts with keys 'id', 'source', 'type', 'timestamp'
        threshold (int): Threshold in seconds to consider alerts as duplicates (default 60s)
        """
    
    sorted_alerts = sorted(alerts, key = lambda x: x['timestamp'])   
    seen = {}
    deduplicated = []              
        
    for alert in sorted_alerts:
        key = (alert['source'], alert['type'])
        last_ts = seen.get(key) # last timestamp in seen for this source/type combo
        
        if last_ts is None or (alert['timestamp'] - last_ts) > threshold:
            deduplicated.append(alert)
            seen[key] = alert['timestamp']
    
    return deduplicated


# Tests
alerts = [
    {'id': 1, 'source': 'sensorA', 'type': 'overheat', 'timestamp': 100},
    {'id': 2, 'source': 'sensorA', 'type': 'overheat', 'timestamp': 130},  # dup
    {'id': 3, 'source': 'sensorA', 'type': 'overheat', 'timestamp': 200},  # outside window
    {'id': 4, 'source': 'sensorB', 'type': 'overheat', 'timestamp': 100},  # different source
]
result = deduplicate_alerts(alerts, 60)
assert [a['id'] for a in result] == [1, 4, 3], f"Got {[a['id'] for a in result]}"
print("All tests passed")


"""You have a list of intelligence reports. Each report has a source, confidence score (0–1), recency score (0–1), and a list of entity mentions. 
Write a function that ranks entities by aggregate importance, where importance is a weighted sum of confidence and recency across all reports mentioning them."""

def aggregate_importance(reports, confidence_weight=0.6, recency_weight=0.4):
    """
    Function to rank entities by aggregate importance based on confidence and recency scores from multiple reports.
    
    Args:
        reports (list): List of dictionaries representing intelligence reports with keys 'source', 'confidence', 'recency', 'entities'
        confidence_weight (float): Weight for confidence score in importance calculation
        recency_weight (float): Weight for recency score in importance calculation
        
    Returns:
        List of tuples (entity, importance_score) sorted by importance_score in descending order.
    """
    
    entity_scores = {}
    
    # Loop through each report and work out how much it contributes to the importance score
    for report in reports:
        report_contribution = confidence_weight * report['confidence'] + recency_weight * report['recency']

        # Loop through each entity in in the report and add the contribution
        for entity in report['entities']:
            entity_scores[entity] = entity_scores.get(entity, 0) + report_contribution
            
    sorted_scores = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
     
    return sorted_scores       

# Tests
reports = [
    {'source': 'A', 'confidence': 0.9, 'recency': 0.8, 'entities': ['Alpha', 'Beta']},
    {'source': 'B', 'confidence': 0.5, 'recency': 0.9, 'entities': ['Alpha']},
    {'source': 'C', 'confidence': 0.1, 'recency': 0.1, 'entities': ['Gamma']},
]
result = aggregate_importance(reports)
assert result[0][0] == 'Alpha'
assert result[-1][0] == 'Gamma'
print("All tests passed:", result)