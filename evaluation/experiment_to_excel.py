import argparse
import gzip
import json
import os
import pandas as pd
import re
import sys
from scipy.stats import ttest_rel
from argparse import ArgumentParser

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False) 
pd.set_option('display.max_colwidth', 256) 

def get_metrics_single_pred(doc):
    result = doc['metrics']
    if 'sampled_metrics' in doc:
        for key in doc['sampled_metrics']:
            result[f"sampled_{key}"] = doc['sampled_metrics'][key]
    return result

def read_predictions(filename):
    result = []
    with gzip.open(filename) as input:
        for line in input:
            doc = json.loads(line)
            metrics = get_metrics_single_pred(doc)
            result.append(metrics)
    return pd.DataFrame(result)


def is_experiment_start(line):
    return line.startswith('evaluating for')

def skip_n_experiments(input_file, experiment_num):
    current_experiment = 0
    while current_experiment < experiment_num:
        line = input_file.readline()
        if is_experiment_start(line):
            current_experiment += 1
            
def get_metrics(line):
    regexp = re.compile(r'[a-zA-Z0-9_]+\: [0-9\.\+\-eE]+')
    result = {}
    for metric_str in regexp.findall(line):
        metric, value = metric_str.split(': ')
        result[metric] = float(value)
    return result


def get_metrics_internal(result, line):
    metrics = line.split(",")
    for metric in metrics:
        name, value = metric.split(":")
        result[name.strip()] = float(value.strip())
    return result 

def parse_experiment(experiment_log):
    current_recommender = None
    result = []
    cnt =0
    metrics = []
    experiment_finished = True
    for line in experiment_log:
            if line.startswith('evaluating ') or line.startswith("!!!!!!!!!   evaluating"):
                current_recommender = line.split(' ')[-1]
                metrics = []
                experiment_finished = False
                epoch = 0
            if 'val_ndcg_at_' in line:
                    epoch += 1
                    epoch_metrics = get_metrics(line)
                    epoch_metrics['epoch'] = epoch
            if 'best_ndcg' in line:
                epoch_metrics = get_metrics_internal(epoch_metrics, line)
                metrics.append(epoch_metrics)

            try:
                experiment_results = json.loads(line)
                experiment_results['model_name'] =  current_recommender
                experiment_results['num_epochs'] = epoch
                experiment_results['metrics_history'] = metrics
                result.append(experiment_results)
                experiment_finished = True
            except:
                pass
    if not experiment_finished:
        experiment_results = {}
        experiment_results['model_name'] =  current_recommender
        experiment_results['metrics_history'] = metrics
        experiment_results['num_epochs'] = epoch
        result.append(experiment_results)
    return result

def get_data_from_logs(logfile, experiment_num):
    current_experiment = 0
    with open(logfile) as input_file:
        skip_n_experiments(input_file, experiment_num)
        experiment_log = []
        for line in input_file:
            if is_experiment_start(line):
                break
            else:
                experiment_log.append(line.strip())
        return parse_experiment(experiment_log)

def main(args):
    interesting_metrics = args.metrics.split(",")
    experiment_filename = os.path.join(args.experiment, "stdout")
    print(experiment_filename)
    if not(os.path.isfile(experiment_filename)):
        raise Exception(f"bad experiment {experiment_filename}")
    experiment_data = get_data_from_logs(experiment_filename,0)
    df = pd.DataFrame(experiment_data).set_index('model_name')
    expxeriment_id = [experiment_filename] * len(df)
    if 'sampled_metrics' in df.columns:
        sampled_metrics_raw = list(df['sampled_metrics'])
        sampled_metrics = []
        for metrics in sampled_metrics_raw:
            if type(metrics) == dict:
                doc = {}
                for metric in metrics:
                    if metric not in interesting_metrics:
                        continue
                    else:
                        doc[f"sampled_{metric}"] = metrics[metric]
                sampled_metrics.append(doc)
            else:
                sampled_metrics.append(dict())
        del(df['sampled_metrics'])
        sampled_metrics_df = pd.DataFrame(sampled_metrics, index=df.index)
    else:
        sampled_metrics_df = pd.DataFrame()
    df = df[interesting_metrics + ['model_build_time']]
    df = pd.concat([sampled_metrics_df, df], axis=1)
    df['experiment_id'] = expxeriment_id
    predictions = {}
    for model in df.index:
        model_prediction_file = os.path.join(args.experiment, "predictions", f"{model}.json.gz")
        print(f"reading predictions from {model_prediction_file}")
        model_predictions = read_predictions(model_prediction_file)
        predictions[model] = model_predictions
    pass

    n_tests = (len(df) - 1) * len(interesting_metrics) * 2 #n-1 significance test per metric, sampled and unsampled 
    for metric in interesting_metrics:
        df = mark_significance(df, metric, n_tests, predictions, args.dec_places)
        sampled_metric = f"sampled_{metric}"
        if sampled_metric in df.columns:
            df = mark_significance(df, sampled_metric, n_tests, predictions, args.dec_places)
    df.model_build_time = df.model_build_time.apply(int)
    if not(args.csv):
        print(df)
    else:
        print(df.to_csv())

def mark_significance(df, metric, n_tests, predictions, dec_places):
    col = df[metric]
    best_model = col.idxmax() 
    res = {}
    res[best_model] =  f"{col[best_model]:{0}.{dec_places}}†" 
    for model in df.index:
        if model != best_model:
            t, pval = ttest_rel(predictions[model][metric], predictions[best_model][metric]) 
            pval *= n_tests #bonferoni correction
            if pval < 0.001:
                res[model] =  f"{col[model]:{0}.{dec_places}}***" 
            elif pval < 0.01:
                res[model] =  f"{col[model]:{0}.{dec_places}}**" 
            elif pval < 0.05:
                res[model] =  f"{col[model]:{0}.{dec_places}}*" 
            else:
                res[model] =  f"{col[model]:{0}.{dec_places}}" 
    res_series = pd.Series(res)
    df[metric] = res_series
    return df





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment")
    parser.add_argument("--metrics")
    parser.add_argument("--dec_places", default=4, type=int)
    parser.add_argument("--csv", type=bool, action=argparse.BooleanOptionalAction )
    args = parser.parse_args()
    print(args)
    main(args)

