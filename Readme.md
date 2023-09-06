# This repository contains code for the paper RecSys'23 "gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling"

**Link to the paper:**[https://arxiv.org/pdf/2308.07192.pdf](https://arxiv.org/pdf/2308.07192.pdf)

If you use this code from the repository, please cite the work: 
```
@inproceedings{petrov2022recencysampling,
  title={gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling},
  author={Petrov, Aleksandr and Macdonald, Craig},
  booktitle={Seventeen ACM Conference on Recommender Systems},
  year={2022}
}
```

# Pytorch version
If you are looking for a pytorch version of gSASRec, please use the official port:  [https://github.com/asash/gSASRec-pytorch/](https://github.com/asash/gSASRec-pytorch/)
The pytorch version is independent of the aprec framework and may be easier to use outside. 

# Environment setup

To set the environment, you can use `Dockerfile` from the `docker` folder and build the image with the help of the docker build command:

```
docker build . -t  gsasrec
```

Alternatively, the `Dockerfile` can be seen as a step-by-step instruction to set up the environment on your machine. 

Our code is based on the `aprec` framework from our recent [reproducibility work](https://github.com/asash/bert4rec_repro), so you can use the original documentation to learn how to use the framework. 

## GSASrec and GBCE info info
**gSASRec** is a SASRec-based sequential recommendation model that utilises more negatives per positive and gBCE loss: 

```math
\begin{align}
     \mathcal{L}^{\beta}_{gBCE} = -\frac{1}{|I_k^-| + 1} \left( \log(\sigma^{\beta}(s_{i^+})) + \sum_{i \in I_k^{-}}\log(1-\sigma(s_i)) \right)
\end{align}
```
where $`i^+`$ is the positive sample, $`I_k^-`$ is the set of negative samples, $`s_i`$ is the model's score for item $`i`$ and $`\sigma`$ is the logistic sigmoid function. 

The $`\beta`$ parameter controls the model calibration level. Note that we do not specify beta directly and infer it from the calibration parameter $`t`$:

```math
\begin{align}
    \beta = \alpha \left(t\left(1 - \frac{1}{\alpha}\right) + \frac{1}{\alpha}\right)
\end{align}
```
Where $`\alpha`$ is the negative sampling rate: $`\frac{`|I_k^-|`}{|I| - 1}`$, and $`|I|`$ is the catalogue size. 


Two models' hyperparameters (in addition to standard SASRec's hyperparameters) are $`k`$ -- the number of negatives per positive, and $`t`$. We recommend using $`k = 256`$ and $`t=0.75`$.  
However, if you want fully calibrated probabilities (e.g., not just to sort items but to use these probabilities as an approximation for CTR), you should set $t=1.0$. In this case, model training will take longer but converge to realistic probabilities (see proofs and experiments in the paper). 

 We do not implement gBCE explicitly. Instead, we use score positive conversion and then use the [standard BCE](losses/bce.py) loss: 
```math
\begin{align}
        \mathcal{L}^{\beta}_{gBCE}(s^+, s^-) =  \mathcal{L}_{BCE}(\gamma(s^+), s^-)
\end{align}
```
where

```math
\begin{align}
    \gamma(s^+)= \log\left(\frac{1}{\sigma^{-\beta}(s^+) - 1}\right)
\end{align}
```

Our SASRec code is based on the original SASRec code. 

The most important code part that you can re-use in other projects: 

```python
  alpha = self.model_parameters.vanilla_num_negatives / (self.data_parameters.num_items - 1)
  t = self.model_parameters.vanilla_bce_t 
  beta = alpha * ((1 - 1/alpha)*t + 1/alpha)

  positive_logits = tf.cast(logits[:, :, 0:1], 'float64') #use float64 to increase numerical stability
  negative_logits = logits[:,:,1:]
  eps = 1e-10
  positive_probs = tf.clip_by_value(tf.sigmoid(positive_logits), eps, 1-eps)
  positive_probs_adjusted = tf.clip_by_value(tf.math.pow(positive_probs, -beta), 1+eps, tf.float64.max)
  to_log = tf.clip_by_value(tf.math.divide(1.0, (positive_probs_adjusted  - 1)), eps, tf.float64.max)
  positive_logits_transformed = tf.math.log(to_log)
  negative_logits = tf.cast(negative_logits, 'float64')
  logits = tf.concat([positive_logits_transformed, negative_logits], -1)
```
The code of our gSASRec model is located in the file [recommenders/sequential/models/sasrec/sasrec.py](recommenders/sequential/models/sasrec/sasrec.py)

### Note that when you use gBCE, the model may require some time to "kick-off" training  and improve the results above simple models like popularity. 

If you observe this pattern, consider increasing early stopping patience - the model eventually will start learning. Alternatively, consider decreasing t in gBCE to make task easier for the model. 

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
