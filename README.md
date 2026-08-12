# FAGEL
Code of the paper: FAGEL: Fast and Accurate Graph Ensemble Learning using Staged Training and Diversified Sampling
The camera-ready paper for ECML PKDD 2026 can be found at: 

## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.

## Data preparation
* Homogeneous datasets for node classification
* Ogb datasets for node classification

These datasets include four medium-scale datasets. Please download them from pytorch geometric [pytorch-geometric-dataset](https://pytorch-geometric.readthedocs.io/en/2.5.2/modules/datasets.html#homogeneous-datasets).
You can download Ogb datasets [Ogbn](https://ogb.stanford.edu/docs/nodeprop/)

---
## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{FAGEL,
  title={FAGEL: Fast and Accurate Graph Ensemble Learning using Staged Training and Diversified Sampling},
  author={Jiajun Shen and Yufei Jin and Xingquan Zhu},
  booktitle={European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD)},
  year      = {2026},
  publisher = {Springer},
  address   = {Naples, Italy}
}
```
If you encounter any issues, please feel free to reach out to me at jshen2024@fau.edu.
