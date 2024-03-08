# Dreaming of Many Worlds: Learning Contextual World Models Aids Zero-Shot Generalization in Reinforcement Learning


## Code

The training and evaluation code for our experiments is at `contextual_mbrl` directory. The `contextual_mbrl/dreamer/envs.py` defines all our context variations for train and eval and  `contextual_mbrl/dreamer/configs.yaml` defines all the configurations and hyperparameters for our runs. The `contextual_mbrl/dreamer/record_dreams.py` allows us to record the extrapolated/counterfactual dreams.

The code in `dreamerv3_compat` is taken from the [fork](https://github.com/Kinds-of-Intelligence-CFI/dreamerv3-compat) which adds gymnasium support to the [official DreamerV3](https://github.com/danijar/dreamerv3) codebase. We changed it to incorporate our cRSSM method. Our changes are localized mainly to `dreamerv3_compat/dreamerv3/nets.py` and `dreamerv3_compat/dreamerv3/agent.py`.

## Setup

### Conda
First setup a miniconda environment.

`conda env create --name c_mbrl --file environment.yml`

## Experiments

### Main experiments

All our main experiments can be replicated by running `sbatch 001_main_experiments.sh` with slurm or modify it according to your training environment. To replicate individual experiments, you have to select one for each of the 4 main options.

1. **Task**
    `carl_classic_cartpole`/`carl_dmc_walker`
2. **Modality** 
    `img` for pixel modality and `obs` for featurized modality
3. **Method** 
    `enc_{$modality}_dec_{$modality}_pgm_ctx` is the cRSSM setting, 
    `enc_{$modality}_dec_{$modality}` is the hidden-context setting, 
    `enc_{$modality}_ctx_dec_{$modality}_ctx` is the concat-context setting
4. **Training context**
    `default`: Only on default context (makes sense to pair only with the hidden-context setting)
    `single_0`: vary the first context (gravity for cartpole and walker)
    `single_0`: vary the second context (length for cartpole and actuator strength for walker)

Then run the following commands your preferred settings and seed to run training followed by evaluation in required context regions.

``` bash
python -m contextual_mbrl.dreamer.train --configs carl $scheme --task $task --env.carl.context $training_context --seed $seed --logdir logs/$experiment_name/$seed --wandb.project '' --run.steps $steps
python -m contextual_mbrl.dreamer.eval --logdir logs/$experiment_name/$seed 
```


### Experts and random policy

To train the experts, evaluate their mean returns and evaluate the models, run `002_cartpole_experts.sh` and `003_walker_experts.sh`.

An example to train cartpole export in gravity 17.64 and length 0.9 is

```
python -m contextual_mbrl.dreamer.train --configs carl enc_obs_dec_obs --task carl_classic_cartpole --env.carl.context specific_0-17.64_1-0.9  --seed 0 --logdir logs/specific/carl_classic_cartpole_specific_0-17.64_1-0.9/0 --wandb.project '' --run.steps 50000
python -m contextual_mbrl.dreamer.eval --logdir logs/specific/carl_classic_cartpole_specific_0-17.64_1-0.9/0
python -m contextual_mbrl.dreamer.eval --logdir logs/specific/carl_classic_cartpole_specific_0-17.64_1-0.9/0 --random_policy True
```

### Dreams
The extrapolated and counterfactual dreams of the trained cartpole models can be obtained using `004_collect_dreams.sh`. For individual recording of the dreams for a given context id (refer `envs.py` for context id to context mapping for different environments) for a given experiment run,

```bash
python -m contextual_mbrl.dreamer.record_dreams --logdir logs/$experiment_name/$seed --ctx_id 1
```

Use the `counterfactual_ctx` flag to provide counterfactual value and record dreams in different true contexts conditioning on this counterfactual value.

```bash
python -m contextual_mbrl.dreamer.record_dreams --logdir logs/$experiment_name/$seed --ctx_id 1 --counterfactual_ctx 1.0
```
## Results

The figures can be plotted from the evaluations for all experiments using the notebooks `plot_analysis.ipynb` and `plot_analysis_rliable.ipynb`