# Active Heterogeneous Graph Domain Adaptation for Cross-network Node Classification(AHGDA)
This repository contains the author's implementation in PyTorch for the paper "Open-set Cross-network Node Classification via Unknown-excluded Adversarial Graph Domain Alignment".

# Environment Requirement
Python: 3.8.20
CUDA: 11.1
torch==1.9.0
numpy==1.24.4
scipy==1.10.1
scikit-learn==1.3.2
networkx==3.1
dgl==0.9.1


For **IMDB1-2**:
```bash
python main_label.py --epoch 200 --datasetS IMDB1 --datasetT IMDB2 --n-fp-layers 2 --n-task-layers 2 --num-hops 4 --num-label-hops 4 --label-feats \
--hidden1 512 --hidden2 512 --dropout 0.5 --input-drop 0.8 --lr 0.0001 --weight-decay 0.05 --batch-size-P 128 --batch-size-A 128 --batch-size-V 64 \
--ad-weightnode 0.1 --ad-weightpath 0.1 --al-weight 0.01 --amp --fine-tune-epochs 10 --seeds 1
```

For **IMDB2-1**:
```bash
python main_label.py --epoch 200 --datasetS IMDB2 --datasetT IMDB1 --n-fp-layers 2 --n-task-layers 2 --num-hops 4 --num-label-hops 4 --label-feats \
--hidden1 512 --hidden2 256 --dropout 0.5 --input-drop 0.8 --lr 0.0001 --weight-decay 0.05 --batch-size-P 128 --batch-size-A 128 --batch-size-V 64 \
--ad-weightnode 0.1 --ad-weightpath 0.1 --al-weight 0.01 --amp --fine-tune-epochs 10 --seeds 1
```

For **dblp11-12**:
```bash
python main_label.py --epoch 200 --datasetS dblp11 --datasetT dblp12 --n-fp-layers 1 --n-task-layers 1 --num-hops 4 --num-label-hops 2 --label-feats \
--hidden1 256 --hidden2 256 --dropout 0.5 --input-drop 0.8 --lr 0.0001 --weight-decay 0 --batch-size-P 512 --batch-size-A 1024 --batch-size-V 64 \
--ad-weightnode 0.1 --ad-weightpath 0.01 --al-weight 0.01 --amp --seeds 1
```

For **dblp12-11**:
```bash
python main_label.py --epoch 200 --datasetS dblp12 --datasetT dblp11 --n-fp-layers 1 --n-task-layers 1 --num-hops 4 --num-label-hops 2 --label-feats \
--hidden1 128 --hidden2 128 --dropout 0.5 --input-drop 0.8 --lr 0.0001 --weight-decay 0 --batch-size-P 512 --batch-size-A 1024 --batch-size-V 64 \
--ad-weightnode 0.1 --ad-weightpath 0.01 --al-weight 0.1 --amp --seeds 1
```

For **dblp11-13**:
```bash
python main_label.py --epoch 200 --datasetS dblp11 --datasetT dblp13 --n-fp-layers 2 --n-task-layers 2 --num-hops 4 --num-label-hops 2 --label-feats \
--hidden1 256 --hidden2 256 --dropout 0.5 --input-drop 0.8 --lr 0.0001 --weight-decay 0 --batch-size-P 512 --batch-size-A 1024 --batch-size-V 64 \
--ad-weightnode 0.1 --ad-weightpath 0.1 --al-weight 0.01 --amp --seeds 1
```

For **dblp13-11**:
```bash
python main_label.py --epoch 200 --datasetS dblp13 --datasetT dblp11 --n-fp-layers 2 --n-task-layers 2 --num-hops 4 --num-label-hops 2 --label-feats \
--hidden1 256 --hidden2 256 --dropout 0.5 --input-drop 0.7 --lr 0.0001 --weight-decay 0 --batch-size-P 256 --batch-size-A 512 --batch-size-V 32 \
--ad-weightnode 0.1 --ad-weightpath 0.01 --al-weight 0.01 --amp --seeds 1
```

For **dblp12-13**
```bash
python main_label.py --epoch 200 --datasetS dblp12 --datasetT dblp13 --n-fp-layers 1 --n-task-layers 2 --num-hops 4 --num-label-hops 2 --label-feats \
--hidden1 256 --hidden2 256 --dropout 0.5 --input-drop 0.8 --lr 0.0001 --weight-decay 0 --batch-size-P 512 --batch-size-A 1024 --batch-size-V 64 \
--ad-weightnode 0.1 --ad-weightpath 0.01 --al-weight 0.01 --amp --seeds 1
```

For **dblp13-12**:
```bash
python main_label.py --epoch 200 --datasetS dblp13 --datasetT dblp12 --n-fp-layers 2 --n-task-layers 2 --num-hops 4 --num-label-hops 2 --label-feats \
--hidden1 256 --hidden2 256 --dropout 0.6 --input-drop 0.5 --lr 0.0001 --weight-decay 0 --batch-size-P 512 --batch-size-A 1024 --batch-size-V 64 \
--ad-weightnode 0.1 --ad-weightpath 0.01 --al-weight 0.01 --amp --seeds 1
```
