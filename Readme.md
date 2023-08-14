# This repository contains code for the paper RecSys'23 "gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling"

If you use this code from repository please cite the work: 
```
@inproceedings{petrov2022recencysampling,
  title={gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling},
  author={Petrov, Aleksandr and Macdonald, Craig},
  booktitle={Seventeen ACM Conference on Recommender Systems},
  year={2022}
}
```



To setup the environment, you can use `Dockerfile` from the `docker` folder and build the image with the help of the docker build command:

```
docker build . -t  gsasrec
```

Alternatively, the `Dockerfile` can be seen as a step-by-step instruction to set up the environment on your machine. 

Our code is based on the `aprec` framework from a recent [reproducibility work](https://github.com/asash/bert4rec_repro), so you can use the original documentation to learn how to use the framework. 

# Runnig experiments
(instruction copied from the original repo)

### 1.  Go to aprec evaluation folder: 
```
cd <your working directory>
cd aprec/evaluation
```

### 2. Run example experiment: 
You need to run `run_n_experiments.sh` with the experiment configuration file. Here is how to do it with an example configuration: 


```
sh run_n_experiments.sh configs/ML1M-bpr-example.py
```
to analyse the results of the latest experiment run 

```
python3 analyze_experiment_in_progress.py
```

### 3. Reproducing experiments from the paper
The config files for  experiments described in the paper are in the `configs/gsasrec/`. 
To run the experiments, please run.

**MovieLens-1M:**

```
sh run_n_experiments.sh configs/gsasrec/ml1m_benchmark.py
```

**Steam:**

```
sh run_n_experiments.sh configs/gsasrec/steam_benchmark.py
```
**Gowalla:**

```
sh run_n_experiments.sh configs/gsasrec/gowalla_benchmark.py
```
### 4. GSASRec code
the code of our **gSASRec** model is located in the file `recommenders/sequential/models/sasrec/sasrec.py`

