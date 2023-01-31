import sys
import gzip
import json
import pandas as pd
import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

prediction_file = sys.argv[1]
data = json.load(gzip.open(prediction_file))
docs = []
for doc in data: 
    docs.append(doc['metrics'])
df = pd.DataFrame(docs)

metrics = []
for metric in (df.columns):
    mean, conf_min, conf_max = mean_confidence_interval(df[metric])
    metric_doc = {'name': metric, 'mean': mean, 'conf_min': conf_min, 'conf_max': conf_max}
    metrics.append(metric_doc)

print(pd.DataFrame(metrics))
    
