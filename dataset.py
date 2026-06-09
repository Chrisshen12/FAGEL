import torch
import torch.nn as nn
from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid, Reddit2
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected, degree,dropout_edge
import random
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from torch.serialization import safe_globals
from torch_geometric.data.data import DataEdgeAttr,DataTensorAttr
import matplotlib.pyplot as plt
from torch_geometric.data.storage import GlobalStorage

import warnings
warnings.filterwarnings("ignore")


def load_Roman(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	dataset = HeterophilousGraphDataset(root='tmp/Roman', name='Roman-empire')
	train_mask = dataset.train_mask[:,0]
	val_mask = dataset.val_mask[:,0]
	test_mask = dataset.test_mask[:,0]
	data = dataset[0]
	data.edge_index = to_undirected(dataset.edge_index)

	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask

	return data

def load_Amazon(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	dataset = HeterophilousGraphDataset(root='tmp/Amazon', name='Amazon-ratings')
	train_mask = dataset.train_mask[:,0]
	val_mask = dataset.val_mask[:,0]
	test_mask = dataset.test_mask[:,0]
	data = dataset[0]
	data.edge_index = to_undirected(dataset.edge_index)

	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask

	return data

def load_Reddit(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)

	dataset = Reddit2(root='tmp/Reddit2')
	data = dataset[0]
	data.edge_index = to_undirected(dataset.edge_index)
	
	return data

def load_Cora(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	# ----------------------------
	# Load Cora dataset
	# ----------------------------
	#dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
	dataset = Planetoid(root='/tmp/Cora', name='Cora')
	#dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
	data = dataset[0]
	#print(data)
	#print(data.y.unique().numel())
	train_mask = dataset.train_mask
	val_mask = dataset.val_mask
	test_mask = dataset.test_mask
	data.edge_index = to_undirected(dataset.edge_index)


	return data

def load_Ogb_a(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	#dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="tmp/Ogb")
	with safe_globals([DataEdgeAttr,DataTensorAttr,GlobalStorage]):
		dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="tmp/Ogba")

	# Get the graph object (PyTorch Geometric Data object)
	data = dataset[0]

	data.edge_index = to_undirected(dataset.edge_index)

	split_idx = dataset.get_idx_split()

	# Load split indices for train/val/test
	num_nodes = data.num_nodes
	train_mask = torch.zeros(num_nodes, dtype=torch.bool)
	val_mask = torch.zeros(num_nodes, dtype=torch.bool)
	test_mask = torch.zeros(num_nodes, dtype=torch.bool)

	train_mask[split_idx["train"]] = True
	val_mask[split_idx["valid"]] = True
	test_mask[split_idx["test"]] = True

	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask
	#print(data.x.shape)
	data.y = dataset.y.view(-1)

	return data

def load_Ogb_p(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	with safe_globals([DataEdgeAttr,DataTensorAttr,GlobalStorage]):
		dataset = PygNodePropPredDataset(name="ogbn-products", root="tmp/Ogbp")

	# Get the graph object (PyTorch Geometric Data object)
	data = dataset[0]

	data.edge_index = to_undirected(dataset.edge_index)

	split_idx = dataset.get_idx_split()

	# Load split indices for train/val/test
	num_nodes = data.num_nodes
	train_mask = torch.zeros(num_nodes, dtype=torch.bool)
	val_mask = torch.zeros(num_nodes, dtype=torch.bool)
	test_mask = torch.zeros(num_nodes, dtype=torch.bool)

	train_mask[split_idx["train"]] = True
	val_mask[split_idx["valid"]] = True
	test_mask[split_idx["test"]] = True

	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask
	#print(data.x.shape)
	data.y = dataset.y.view(-1)

	return data

def load_Ogb_m(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	
	with safe_globals([DataEdgeAttr,DataTensorAttr,GlobalStorage]):
		dataset = PygNodePropPredDataset(name="ogbn-papers100M", root="tmp/Ogbm")

	# Get the graph object (PyTorch Geometric Data object)
	data = dataset[0]

	data.edge_index = to_undirected(dataset.edge_index)

	split_idx = dataset.get_idx_split()

	# Load split indices for train/val/test
	num_nodes = data.num_nodes
	train_mask = torch.zeros(num_nodes, dtype=torch.bool)
	val_mask = torch.zeros(num_nodes, dtype=torch.bool)
	test_mask = torch.zeros(num_nodes, dtype=torch.bool)

	train_mask[split_idx["train"]] = True
	val_mask[split_idx["valid"]] = True
	test_mask[split_idx["test"]] = True

	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask
	#print(data.x.shape)
	data.y = dataset.y.view(-1)

	return data

	return data