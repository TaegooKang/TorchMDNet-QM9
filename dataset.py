import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def get_qm9_loader(split='train', batch_size=32, shuffle=True):
    """
    Get QM9 dataset loader:
        Args:
            split: 'train', 'test' 
    """
    assert split in ['train', 'test'], "split should be in ['train', 'test']"
    data = torch.load(f'data/qm9_{split}_data.pt')
    data_list = []
        
    if split == 'train':
        y = data['mu'] # target 값: dipole moment value
        num_train = int(len(data['x']) * 0.8) 

    num_nodes = data['num_atoms']  # 분자 내 원자 수(==그래프의 노드 갯수)
    num_edges = data['num_bonds']  # 분자 내 결합 수(==그래프의 엣지 갯수)
    coords = data['x']  # 각 원자의 3d 좌표 값
    one_hot = data['one_hot'] # 노드 피쳐 one-hot embedding, ex) H -> [True, False, False, False, False]
    atomic_numbers = data['atomic_numbers']  # 각 원자의 원자번호
    edge = data['edge']  # 엣지 인덱스와 결합 종류

    print(f'Start {split} data processing....')
    for i in range(len(num_nodes)):
        if split == 'train':
            y_s = torch.tensor(y[i], dtype=torch.float)
        num_node_s = num_nodes[i]
        num_edge_s = num_edges[i]
        coord = torch.tensor(coords[i][:num_node_s])  
        atomic_num = torch.tensor(atomic_numbers[i][:num_node_s, :], dtype=torch.long).squeeze()
        edge_index1 = torch.tensor(edge[i][:num_edge_s, :2], dtype=torch.long).t()
        edge_index2 = torch.index_select(edge_index1, 0, torch.tensor([1,0]))
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        edge_attr = torch.tensor(edge[i][:num_edge_s, 2], dtype=torch.long)
        edge_attr = F.one_hot(torch.cat([edge_attr, edge_attr], dim=0), num_classes=4).float()
        x = torch.tensor(one_hot[i][:num_node_s], dtype=torch.float)
       
        if split == 'train':
            mol = Data(pos=coord, x=x, z=atomic_num, y=y_s, edge_index=edge_index, edge_attr=edge_attr)
        else:
            mol = Data(pos=coord, x=x, z=atomic_num, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(mol)
        
    print(f'Processing is finished!! QM9 {split}_loader init.')
    
    if split == 'train':
        qm9_train_loader = DataLoader(data_list[:num_train], batch_size=batch_size, shuffle=shuffle)
        qm9_valid_loader = DataLoader(data_list[num_train:], batch_size=batch_size, shuffle=False)
        return qm9_train_loader, qm9_valid_loader
    
    else:
        qm9_test_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        return qm9_test_loader

