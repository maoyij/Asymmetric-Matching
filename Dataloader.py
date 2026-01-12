"""
Data loader for abdominal lymph node matching.

This module implements the dataset class for loading lymph node data from CT scans,
including 3D volumes, 2D projections, and graph structures for graph matching.
"""

import gc
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data

from src.build_graphs import build_graphs
from src.factorize_graph_matching import kronecker_sparse
from src.sparse_torch import CSCMatrix3d, CSRMatrix3d
from config import cfg

# Average coordinates for normalization (should be configurable)
AVG_X = 509.3768472906404
AVG_Y = 356.6576354679803
AVG_Z = 222.37192118226602


def To2d(data):
    """
    Extract 9 different 2D views from a 3D volume.
    
    According to the paper, nine 2D views are extracted:
    - Three orthogonal views (axial, coronal, sagittal)
    - Six diagonal/oblique views
    
    Args:
        data: 3D numpy array of shape (z, y, x)
        
    Returns:
        image_list: List of 9 2D views, each with shape (1, height, width)
    """
    z, y, x = data.shape
    
    # Three orthogonal views
    horizontal_slice = data[round(z / 2), :, :]  # Axial
    vertical_slice = data[:, round(y / 2), :]    # Coronal
    depth_slice = data[:, :, round(x / 2)]       # Sagittal
    
    # Six diagonal/oblique views
    diagonal_slice = np.diagonal(data, axis1=0, axis2=1)
    diagonal_flip_slice = np.diagonal(np.fliplr(data), axis1=0, axis2=1)
    diagonal2_slice = np.diagonal(data, axis1=1, axis2=2)
    diagonal2_flip_slice = np.diagonal(np.fliplr(data), axis1=1, axis2=2)
    diagonal3_slice = np.diagonal(data, axis1=2, axis2=0)
    diagonal3_flip_slice = np.diagonal(np.fliplr(data), axis1=2, axis2=0)
    
    # Stack all views
    views = [
        horizontal_slice, vertical_slice, depth_slice,
        diagonal_slice, diagonal2_slice, diagonal3_slice,
        diagonal_flip_slice, diagonal2_flip_slice, diagonal3_flip_slice
    ]
    
    image_list = [np.expand_dims(view, axis=0) for view in views]
    return image_list


class MLNMGraph(torch.utils.data.Dataset):
    """
    Dataset class for Multi-Lymph Node Matching.
    
    Loads lymph node data from CT scans including:
    - 3D node volumes
    - 2D projections (9 views per node)
    - Graph structures for graph matching
    - Ground truth matching matrices
    """
    
    def __init__(self, records_path, config, phase='train', data_root=''):
        """
        Initialize the dataset.
        
        Args:
            records_path: Path to the case list file
            config: Configuration dictionary
            phase: 'train', 'val', or 'test'
            data_root: Root directory for data files
        """
        assert phase in ['train', 'val', 'test'], f"Invalid phase: {phase}"
        
        self.phase = phase
        self.data_root = data_root
        
        # Data paths (should be configurable via config file)
        self.img_root = config.get('img_root', './data/npy')
        self.gt_root = config.get('gt_root', './data/gt')
        self.gt_encoder = config.get('gt_encoder', './data/encode_gt')
        self.gt_chao = config.get('gt_chao', './data/gt_chao')
        self.node_auto_root = config.get('node_auto_root', './data/npy_node_auto')
        
        # Load case list
        self.caselist = np.load(records_path, allow_pickle=True)
    
    def __len__(self):
        """Return the number of cases in the dataset."""
        return len(self.caselist)
    
    @staticmethod
    def to_pyg_graph(A, P):
        """
        Convert adjacency matrix and node coordinates to PyTorch Geometric graph.
        
        Args:
            A: Adjacency matrix
            P: Node coordinates
            
        Returns:
            pyg_graph: PyTorch Geometric Data object
        """
        rescale = max(cfg.PROBLEM.RESCALE)
        
        # Compute edge features based on coordinate differences
        edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5
        edge_index = np.nonzero(A)
        edge_attr = edge_feat[edge_index]
        edge_attr = np.clip(edge_attr, 0, 1)
        
        assert (edge_attr > -1e-5).all(), "Edge attributes should be non-negative"
        
        pyg_graph = pyg.data.Data(
            x=torch.tensor(P / rescale, dtype=torch.float32),
            edge_index=torch.tensor(np.array(edge_index), dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )
        return pyg_graph
    
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            data_dict: Dictionary containing all data for graph matching
        """
        # Parse case names
        case_str = self.caselist[idx]
        t1_case_name = case_str.split(':')[0]
        t2_case_name = case_str.split(':')[1]
        patient_id = t1_case_name.split('_')
        
        # Load ground truth matching matrices
        gt_path = os.path.join(self.gt_root, f'{case_str}_gt.npy')
        match_gt_label = np.load(gt_path, allow_pickle=True)
        
        gt_encode_path = os.path.join(self.gt_encoder, f'{case_str}_gt.npy')
        match_gt_encoder = np.load(gt_encode_path, allow_pickle=True)
        
        gt_chao_path = os.path.join(self.gt_chao, f'{case_str}_gt.npy')
        match_gt_chao = np.load(gt_chao_path, allow_pickle=True)
        
        # Load node labels
        template_labels = np.load(
            os.path.join(self.img_root, f'{t1_case_name}_label.npy'), 
            allow_pickle=True
        )
        searching_labels = np.load(
            os.path.join(self.img_root, f'{t2_case_name}_label.npy'), 
            allow_pickle=True
        )
        
        # Load reference images for coordinate normalization
        template_clean = np.load(os.path.join(self.img_root, f'{t1_case_name}_clean.npy'))
        search_clean = np.load(os.path.join(self.img_root, f'{t2_case_name}_clean.npy'))
        
        # Process T1 nodes
        P1 = []
        img1 = []
        t1_2d_list = []
        n1 = 0
        
        for i, label in enumerate(template_labels):
            # Load 3D node volume
            save_filename = os.path.join(self.node_auto_root, f'{t1_case_name}_{i}.npy')
            all_node = np.load(save_filename, allow_pickle=True)
            img1.append(all_node)
            
            # Extract 2D projections
            t1_2d_list.append(To2d(all_node[0, :, :, :]))
            
            # Normalize coordinates
            P1.append([
                label[2] / template_clean.shape[3] * AVG_X,
                label[1] / template_clean.shape[2] * AVG_Y,
                label[0] / template_clean.shape[1] * AVG_Z
            ])
            n1 += 1
        
        img1 = np.array(img1)
        P1 = np.array(P1)
        
        # Process T2 nodes
        P2 = []
        img2 = []
        t2_2d_list = []
        n2 = 0
        
        for i, label in enumerate(searching_labels):
            # Load 3D node volume
            save_filename = os.path.join(self.node_auto_root, f'{t2_case_name}_{i}.npy')
            all_node = np.load(save_filename, allow_pickle=True)
            img2.append(all_node)
            
            # Extract 2D projections
            t2_2d_list.append(To2d(all_node[0, :, :, :]))
            
            # Normalize coordinates
            P2.append([
                label[2] / search_clean.shape[3] * AVG_X,
                label[1] / search_clean.shape[2] * AVG_Y,
                label[0] / search_clean.shape[1] * AVG_Z
            ])
            n2 += 1
        
        img2 = np.array(img2)
        P2 = np.array(P2)
        
        # Build graphs
        A1, G1, H1, e1 = build_graphs(P1, n1, stg='fc', sym=True)
        A2, G2, H2, e2 = build_graphs(P2, n2, stg='fc', sym=True)
        
        # Convert to PyTorch Geometric graphs
        pyg_graph1 = self.to_pyg_graph(A1, P1)
        pyg_graph2 = self.to_pyg_graph(A2, P2)
        
        # Prepare data dictionary
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        data_dict = {
            'Ps': [torch.Tensor(x).to(device).requires_grad_(True) for x in [P1, P2]],
            'ns': [torch.tensor(x).to(device) for x in [n1, n2]],
            'ns_encoder': [torch.tensor(x).to(device) for x in [n1 + n2, n1 + n2]],
            'es': [torch.tensor(x).to(device) for x in [e1, e2]],
            'gt_perm_mat': torch.from_numpy(match_gt_label).to(device),
            'gt_encoder_mat': torch.from_numpy(match_gt_encoder).to(device),
            'gt_chao_path': torch.from_numpy(match_gt_chao).to(device),
            'Gs': [torch.Tensor(x).to(device).requires_grad_(True) for x in [G1, G2]],
            'Hs': [torch.Tensor(x).to(device).requires_grad_(True) for x in [H1, H2]],
            'As': [torch.Tensor(x).to(device).requires_grad_(True) for x in [A1, A2]],
            'pyg_graphs': [pyg_graph1.to(device), pyg_graph2.to(device)],
            'id_list': len(patient_id),
            'imgs': [torch.Tensor(img).to(device).requires_grad_(True) for img in [img1, img2]],
            'imgs_2d': [torch.Tensor(img).to(device).requires_grad_(True) for img in [t1_2d_list, t2_2d_list]]
        }
        
        if self.phase == "val":
            data_dict['batch_size'] = 1
        
        # Clean up
        del match_gt_label, match_gt_encoder, img1, img2
        del pyg_graph1, pyg_graph2, t1_2d_list, t2_2d_list
        gc.collect()
        
        return data_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    
    This function handles batching of variable-sized graphs and computes
    Kronecker products for efficient graph matching.
    
    Args:
        data: List of data dictionaries
        
    Returns:
        ret: Batched data dictionary
    """
    def pad_tensor(inp):
        """Pad tensors to the same size for batching."""
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        
        max_shape = np.array(max_shape)
        padded_ts = []
        
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))
        
        return padded_ts
    
    def stack(inp):
        """Recursively stack nested structures."""
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Keys mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == pyg.data.Data:
            ret = pyg.data.Batch.from_data_list(inp)
        elif type(inp[0]) == str:
            ret = inp
        elif type(inp[0]) == tuple:
            ret = inp
        elif type(inp[0]) == int:
            ret = inp
        else:
            raise ValueError(f'Cannot handle type {type(inp[0])}')
        return ret
    
    ret = stack(data)
    
    # Compute Kronecker products for graph matching
    if 'Gs' in ret and 'Hs' in ret:
        G1, G2 = ret['Gs']
        H1, H2 = ret['Hs']
        
        sparse_dtype = np.float16 if cfg.FP16 else np.float32
        
        if G1.shape[0] > 1:
            # Batch processing
            KGHs_sparse = []
            for b in range(G1.shape[0]):
                K1G = [
                    kronecker_sparse(x.cpu().detach(), y.cpu().detach()).astype(sparse_dtype)
                    for x, y in zip(G2[b].unsqueeze(0), G1[b].unsqueeze(0))
                ]
                K1H = [
                    kronecker_sparse(x.cpu().detach(), y.cpu().detach()).astype(sparse_dtype)
                    for x, y in zip(H2[b].unsqueeze(0), H1[b].unsqueeze(0))
                ]
                
                K1G_sparse = CSCMatrix3d(K1G)
                K1H_sparse = CSCMatrix3d(K1H).transpose()
                KGHs_sparse.append((K1G_sparse.indices, K1H_sparse.indices))
            
            ret['KGHs_sparse'] = KGHs_sparse
        else:
            # Single sample processing
            K1G = [
                kronecker_sparse(x.cpu().detach(), y.cpu().detach()).astype(sparse_dtype)
                for x, y in zip(G2, G1)
            ]
            K1H = [
                kronecker_sparse(x.cpu().detach(), y.cpu().detach()).astype(sparse_dtype)
                for x, y in zip(H2, H1)
            ]
            
            K1G_sparse = CSCMatrix3d(K1G)
            K1H_sparse = CSCMatrix3d(K1H).transpose()
            ret['KGHs_sparse'] = [(K1G_sparse.indices, K1H_sparse.indices)]
        
        # Also compute dense Kronecker products
        K1G = [
            kronecker_sparse(x.cpu().detach(), y.cpu().detach()).astype(sparse_dtype)
            for x, y in zip(G2, G1)
        ]
        K1H = [
            kronecker_sparse(x.cpu().detach(), y.cpu().detach()).astype(sparse_dtype)
            for x, y in zip(H2, H1)
        ]
        
        K1G = CSRMatrix3d(K1G)
        K1H = CSRMatrix3d(K1H).transpose()
        ret['KGHs'] = K1G, K1H
    
    ret['batch_size'] = len(data)
    return ret
