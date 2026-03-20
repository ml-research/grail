# GRAIL: Grounding Relational Concepts in Neuro-Symbolic Reinforcement Learning

Henri Johannes Rößler<sup>1</sup>, [Hikaru Shindo](https://www.hikarushindo.com/)<sup>2</sup>

<sup>1</sup>Department of Computer Science, TU Darmstadt, Germany
<sup>2</sup>Artificial Intelligence and Machine Learning Lab, TU Darmstadt, Germany

Contact: henri.roessler@tu-darmstadt.de

Neuro-symbolic agents rely on logical rules to infer their actions, which often requires knowledge about how objects are related to each other. Understanding concepts such as *left of* or *nearby* is therefore essential for solving abstract tasks. In existing systems, such relations are typically defined by human experts, which limits extensibility since the meaning of a concept can vary across different environments.

**GRAIL** introduces a framework that **grounds relational concepts through interaction with the environment**. We further utilize large language models to provide additional weak supervision and to complement sparse reward signals. Empirical evaluations on the ATARI environments *Kangaroo* and *Seaquest* demonstrate that our agents match, and in some cases exceed, the performance of logic agents with hand-crafted relational concepts.

---

## Installation

### Using Docker (Recommended)

```bash
docker build -t grail:base .
```

### Local Installation

1. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Install the logic reasoning libraries:
    ```bash
    cd nsfr && pip install -e . && cd ..
    cd nudge && pip install -e . && cd ..
    ```

3. *(Optional)* Install NEUMANN for memory-efficient reasoning (required for highly-parallelized environments, e.g., 512 envs in Seaquest):
    ```bash
    cd neumann && pip install -e . && cd ..
    ```
    This requires [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and its dependencies:
    ```bash
    pip install torch-geometric torch-sparse torch-scatter
    ```
    For GPU support with CUDA 12.4:
    ```bash
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
    ```

4. Install difflogic:
    ```bash
    pip install -e third_party/difflogic
    ```

---

## Usage

### Training

**Stage 1 — Train valuation functions** (learns relational concept grounding with LLM weak supervision):
```bash
python train_valuation.py --env-name kangaroo --num-steps 128 --num-envs 5 --track
```

**Stage 2 — Train the full agent** (joint symbolic + neural policy):
```bash
python train_blenderl.py --env-name seaquest --joint-training --num-steps 128 --num-envs 5 --gamma 0.99
```

**Baselines:**
```bash
# NUDGE (symbolic-only)
python train_nudge.py --env-name kangaroo --num-steps 128 --num-envs 5

# Neural PPO
python train_neuralppo.py --env-name kangaroo --num-steps 128 --num-envs 5
```

### Evaluation

```bash
python eval_valuation.py --env-name kangaroo
python evaluate.py --env-name seaquest --agent-path <path-to-checkpoint>
```

### Interactive Play

```bash
python play_gui.py --env-name kangaroo --agent-path <path-to-checkpoint>
```

Use `--track` on any training script to log to [Weights & Biases](https://wandb.ai/).

---

## Supported Environments

| Environment | Description |
|---|---|
| Kangaroo | Platformer with ladders, monkeys, and coconuts |
| Seaquest | Underwater navigation with divers, enemies, and missiles |
| Freeway | Road crossing with traffic |
| Asterix | Navigation and collection |
| Donkey Kong | Platformer with barrels and ladders |
| Getout | Grid-based navigation |
| Loot | Object collection |
| Threefish | Multi-agent fish environment |

---

## Adding New Environments

Create a new directory under `in/envs/<env_name>/` containing:

- **Logic state extraction** — translates raw environment states into logic representations
- **Valuation functions** — each relation (e.g., `closeby`) maps to a differentiable function computing the probability that the relation holds
- **Action mapping** — maps action-predicates predicted by the agent to environment actions

See `in/envs/freeway/` for a minimal example.

---

## Project Structure

```
├── blendrl/           # Core BlendRL framework (agents, evaluator, explainer)
├── cleanrl/           # Clean RL baselines
├── nsfr/              # Neural-Symbolic Forward Reasoner
├── nudge/             # NUDGE symbolic RL framework
├── neumann/           # Memory-efficient graph-based reasoner
├── valuation/         # Learned valuation functions module
├── in/
│   ├── envs/          # Environment definitions and logic rules
│   ├── config/        # Hyperparameter configurations
│   └── prompts/       # LLM prompt templates for proxy function generation
├── env_src/           # Environment source code
├── third_party/       # External dependencies (difflogic)
├── train_valuation.py # Stage 1: valuation function training
├── train_blenderl.py  # Stage 2: full BlendRL agent training
├── train_nudge.py     # NUDGE baseline training
├── train_neuralppo.py # Neural PPO baseline training
├── eval_valuation.py  # Valuation evaluation
├── sim_valuation.py   # Valuation simulation
└── play_gui.py        # Interactive GUI for trained agents
```

---

## Citation

```bibtex
@article{roessler2025grail,
  title={Learning Relational Concepts in Neuro-Symbolic Reinforcement Learning},
  author={Rößler, Henri Johannes and Shindo, Hikaru},
  year={2025}
}
```

---

## Acknowledgements

GRAIL builds upon [BlendRL](https://github.com/ml-research/blendrl) (ICLR 2025), [NUDGE](https://github.com/ml-research/NUDGE), and [NSFR](https://github.com/ml-research/nsfr).
