# ABPO

## Installation

```bash
pip install -e .
pip install vllm==0.8.2

cd ./utils/latex2sympy
pip install -e .
```

## Training

```bash
bash train_ABPO.sh
```

## Converting Weights

```bash
bash convert.sh
```

## Evaluation

```bash
bash eval.sh
```
