# contextual_mbrl

Do world models actually model the world :p

## Setup

### VScode
Install the recommended extensions when you open the project


### Conda

`conda env create --name c_mbrl --file environment.yml`


## Experiments


### All main experiments

All our main experiments can be replicated by running `sbatch train_jobs.sh` with slurm. To replicate individual experiments, you have to select one for each of the 4 main options.
*Task*: `carl_classic_cartpole`/`carl_dmc_walker`
*Modality*: `img` for pixel modality and `obs` for featurized modality
*Method*:
`enc_{$modality}_dec_{$modality}_pgm_ctx` is the cRSSM setting
`enc_{$modality}_dec_{$modality}` is the hidden-context setting
`enc_{$modality}_ctx_dec_{$modality}_ctx` is the concat-context setting
*Training context*:
`default`: Only on default context (makes sense to pair only with the hidden-context setting)
`single_0`: vary the first context (gravity for cartpole and walker)
`single_0`: vary the second context (length for cartpole and actuator strength for walker)

Then run the following with your preferred settings and seeds.
``` bash

python -m contextual_mbrl.dreamer.train --configs carl $scheme --task $task --env.carl.context $training_context --seed $seed --logdir logs/$experiment_name/$seed --wandb.project '' --run.steps $steps
python -m contextual_mbrl.dreamer.eval --logdir logs/$experiment_name/$seed 
```

### Dreams
The extrapolated and counterfactual dreams of the trained cartpole models can be obtained using `schedule_dreams.sh`. For individual recording of the dreams for a given context id (refer `envs.py` for context id to context mapping for different environments) for a given experiment run,
```bash
python -m contextual_mbrl.dreamer.record_dreams --logdir logs/$experiment_name/$seed --ctx_id 1
```
Use the `counterfactual_ctx` flag to provide counterfactual value and record dreams in different true contexts conditioning on this counterfactual value.
```bash
python -m contextual_mbrl.dreamer.record_dreams --logdir logs/$experiment_name/$seed --ctx_id 1 --counterfactual_ctx 1.0
```

### Experts and random policy

To train the experts, evaluate their mean returns and evaluate the models, run `schedule_specific_jobs_carpole/walker.sh`.

An example to train cartpole export in gravity 17.64 and length 0.9 is
```
python -m contextual_mbrl.dreamer.train --configs carl enc_obs_dec_obs --task carl_classic_cartpole --env.carl.context specific_0-17.64_1-0.9  --seed 0 --logdir logs/specific/carl_classic_cartpole_specific_0-17.64_1-0.9/0 --wandb.project '' --run.steps 50000
python -m contextual_mbrl.dreamer.eval --logdir logs/specific/carl_classic_cartpole_specific_0-17.64_1-0.9/0
python -m contextual_mbrl.dreamer.eval --logdir logs/specific/carl_classic_cartpole_specific_0-17.64_1-0.9/0 --random_policy True
```

## Results

The figures can be plotted from the evaluations for all experiments using the notebooks `plot_analysis.ipynb` and `plot_analysis_rliable.ipynb`