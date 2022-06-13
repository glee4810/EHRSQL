import re

def post_process_sql(query,
                     current_time="2105-12-31 23:59:00",
                     precomputed_dict={
                                'temperature': (35.5, 38.1),
                                'sao2': (95.0, 100.0),
                                'heart rate': (60.0, 100.0),
                                'respiration': (12.0, 18.0),
                                'systolic bp': (90.0, 120.0),
                                'diastolic bp':(60.0, 90.0),
                                'mean bp': (60.0, 110.0)
                            }):
    if "current_time" in query:
        query = query.replace("current_time", f"'{current_time}'")
    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
        if len(vital_name_list)==1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in precomputed_dict:
                vital_range = precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")
    query = query.replace("''", "'").replace('< =', '<=')
    query = query.replace("%y", "%Y").replace('%j', '%J')
    return query

