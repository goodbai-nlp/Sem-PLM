# Sem-PLM

Code for COLING2022 paper "Graph Pre-training for AMR Parsing and Generation". You may find our paper [here](https://aclanthology.org/2022.coling-1.49/).

## Requirements

+ python==3.8
+ pytorch==1.8
+ pytorch-lightning==1.6.0
+ transformers==4.10.0
+ datasets==2.0.0

We train the model using 8 Tesla V100 GPU.
## Training

```
cd src
bash train-3task-roberta.sh
```

## Pre-trained Models
todo

## References

```
@inproceedings{bai-etal-2022-semantic,
    title = "Semantic-based Pre-training for Dialogue Understanding",
    author = "Bai, Xuefeng  and
      Song, Linfeng  and
      Zhang, Yue",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.49",
    pages = "592--607"
}
```