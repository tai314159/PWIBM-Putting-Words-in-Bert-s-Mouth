# Putting Words in BERT's Mouth: Navigating Contextualized Vector Spaces with Pseudowords

This repository contains the code of our paper [Putting Words in BERT's Mouth: Navigating Contextualized Vector Spaces with Pseudowords](https://arxiv.org/abs/2109.11491) (EMNLP 2021).

## MaPP Dataset 
The dataset can be found [here](https://github.com/tai314159/PWIBM-Putting-Words-in-Bert-s-Mouth/tree/main/data/csv). 
It is devided to 3 portions (as we describe in our paper). 

## Get Pseudowords 
To get the pseudoword vectors, run the code --> get_pseudowords.py using the data (queries) we provide [here](https://github.com/tai314159/PWIBM-Putting-Words-in-Bert-s-Mouth/blob/main/data/queries/single_target/MaPP_all.txt), or data of the same format.

## Citation
Please cite our paper if you found the resources in this repository useful.

```bibtex
@inproceedings{karidi2021putting,
      title={Putting Words in BERT's Mouth: Navigating Contextualized Vector Spaces with Pseudowords}, 
      author={Taelin Karidi and Yichu Zhou and Nathan Schneider and Omri Abend and Vivek Srikumar},
      year={2021},
      eprint={2109.11491},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
}
``` 