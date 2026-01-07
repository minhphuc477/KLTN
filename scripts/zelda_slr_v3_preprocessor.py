import numpy as np
import torch
import networkx as nx
import re
from torch.utils.data import Dataset

class ZeldaSLRv3Dataset(Dataset):
    def __init__(self, txt_path, dot_path, room_dims=(11, 16)):
        self.room_h, self.room_w = room_dims
        self.tiles_map = self._get_slr_v3_mapping()
        
        # 1. Parse 'Thể xác' (.txt)
        self.full_grid = self._load_txt(txt_path)
        
        # 2. Parse 'Linh hồn' (.dot)
        self.nodes, self.edges, self.graph = self._parse_dot(dot_path)
        
        # 3. Mapping Tọa độ (Vị trí phòng trong file txt tương ứng với Node ID)
        # Lưu ý: Đây là tọa độ dựa trên Grid xác thực tao đã làm cho mày
        self.room_mapping = {
            7: (6, 2), 8: (5, 2), 5: (5, 1), 6: (5, 3), 0: (5, 4),
            14: (4, 4), 2: (4, 0), 10: (3, 4), 13: (3, 3), 12: (3, 2),
            16: (3, 1), 18: (3, 0), 3: (2, 4), 9: (2, 3), 1: (2, 2),
            17: (2, 1), 15: (1, 2), 11: (0, 2)
        }
        
        # 4. Tính toán TPE (Laplacian Eigenvectors)
        self.tpe = self._compute_tpe(self.graph)

    def _get_slr_v3_mapping(self):
        """Định nghĩa bảng mã SLR v3 để giải quyết xung đột ký tự"""
        return {
            'F': 1, 'W': 2, 'D': 3, 'M': 4, 'B': 5, 'P': 6, 
            'k': 10,  # Key (Logic-aware)
            's': 11,  # Start
            't': 12,  # Triforce
            'b': 13,  # Boss
            'I_logic': 14, # Key Item
            'S_logic': 15, # Switch
            '-': 0    # Void
        }

    def _load_txt(self, path):
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return lines

    def _parse_dot(self, path):
        with open(path, 'r') as f:
            content = f.read()
        
        G = nx.DiGraph()
        nodes = {}
        # Extract Nodes
        node_matches = re.findall(r'(\d+) \[label="(.*?)"\]', content)
        for nid, label in node_matches:
            nid = int(nid)
            nodes[nid] = label.split(',')
            G.add_node(nid, label=label)
            
        # Extract Edges
        edge_matches = re.findall(r'(\d+) -> (\d+) \[label="(.*?)"\]', content)
        edges = []
        for u, v, label in edge_matches:
            u, v = int(u), int(v)
            edges.append((u, v, label))
            G.add_edge(u, v, label=label)
            
        return nodes, edges, G

    def _compute_tpe(self, G, k=4):
        """Tính toán Topological Positional Encoding (GPS cho hầm ngục)"""
        L = nx.laplacian_matrix(G.to_undirected()).astype(float).todense()
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Lấy k eigenvectors đầu tiên (bỏ qua cái đầu tiên vì nó bằng 0)
        return torch.from_numpy(eigenvectors[:, 1:k+1]).float()

    def get_room_tensor(self, node_id):
        """Cắt và nhuộm mã SLR v3 cho từng phòng cụ thể"""
        row_idx, col_idx = self.room_mapping[node_id]
        start_r = row_idx * self.room_h
        start_c = col_idx * self.room_w
        
        raw_room = [line[start_c:start_c + self.room_w] 
                    for line in self.full_grid[start_r:start_r + self.room_h]]
        
        # Nhuộm mã SLR v3
        node_labels = self.nodes[node_id]
        room_tensor = torch.zeros((self.room_h, self.room_w), dtype=torch.long)
        
        for r in range(self.room_h):
            for c in range(self.room_w):
                char = raw_room[r][c]
                tile_id = self.tiles_map.get(char, 0)
                
                # Logic Injection: Nếu phòng có 'k' trong DOT, biến tile 'I' hoặc 'M' thành Key
                if char == 'M' and 'k' in node_labels:
                    tile_id = self.tiles_map['k']
                elif char == 'S' and 's' in node_labels:
                    tile_id = self.tiles_map['s']
                
                room_tensor[r, c] = tile_id
                
        return room_tensor

    def __len__(self):
        return len(self.room_mapping)

    def __getitem__(self, idx):
        node_id = list(self.room_mapping.keys())[idx]
        room_tiles = self.get_room_tensor(node_id)
        room_tpe = self.tpe[node_id]
        
        return {
            "node_id": node_id,
            "tiles": room_tiles,
            "tpe": room_tpe,
            "labels": self.nodes[node_id]
        }

# --- TEST SCRIPT ---
# dataset = ZeldaSLRv3Dataset("tloz1_1.txt", "tloz1_1.dot")
# print(f"Loaded {len(dataset)} rooms.")
# sample = dataset[0]
# print(f"Room {sample['node_id']} tiles shape: {sample['tiles'].shape}")