import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import torch
from torch_geometric.data import Data

def prepare_data(data_platform: str = "10XVisium", sample_number: int = 151507, split_number: int = 0,
                 preprocessed: bool = True, impute_gene_type: str = "all",
                 use_spatial_graph: bool = False, spatial_graph_type: str = "radius",
                 spatial_graph_radius_cutoff: int = 150, spatial_graph_knn_cutoff: int = 10,
                 spatial_graph_optimize: str = None, spatial_graph_cluster_resolution: float = 1,
                 use_gene_graph: bool = False, gene_graph_knn_cutoff: int = 10, 
                 gene_graph_num_high_var_genes: int = 100, gene_graph_pca_select: bool = True, 
                 gene_graph_pca_num: int = 50, use_morphology_graph: bool = False, morphology_knn_cutoff: int = 5, 
                 use_heterogeneous_graph: bool = False):

    if data_platform == "10XVisium":
        assert (use_spatial_graph or use_gene_graph or use_morphology_graph) == True
        if use_heterogeneous_graph:
            assert sum([use_spatial_graph, use_gene_graph, use_morphology_graph]) > 1  

        data_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC/"
        adata = sc.read_h5ad(f"{data_dir}/Preprocessed/{sample_number}/filtered_adata.h5ad")

        # 读取 ground truth 信息
        with open(f"{data_dir}/Preprocessed/{sample_number}/ground_truth_layer.pkl", 'rb') as file:
            ground_truth_layer = pickle.load(file)
        adata.obs['Ground Truth'] = list(ground_truth_layer)

        # 读取 `val_mask` 和 `test_mask`
        val_mask = np.load(f"{data_dir}/Preprocessed/{sample_number}/split_{split_number}_val_mask.npz")['arr_0']
        test_mask = np.load(f"{data_dir}/Preprocessed/{sample_number}/split_{split_number}_test_mask.npz")['arr_0']
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

        # **创建 `X_masked` 版本**
        X_dense = np.array(adata.X.todense().astype(np.float32))
        X_masked = X_dense.copy()
        X_masked[val_mask.numpy() == 1] = 0  
        X_masked[test_mask.numpy() == 1] = 0  
        
        spatial_graph_coor = pd.DataFrame(adata.obsm['spatial'])
        spatial_graph_coor.index = adata.obs.index
        spatial_graph_coor.columns = ['imagerow', 'imagecol']
        id_cell_trans = dict(zip(range(spatial_graph_coor.shape[0]), np.array(spatial_graph_coor.index)))
        cell_to_index = {id_cell_trans[idx]: idx for idx in id_cell_trans}
        # For spatial graph
        if use_spatial_graph:
            if spatial_graph_type == "radius":
                nbrs = NearestNeighbors(radius=spatial_graph_radius_cutoff).fit(spatial_graph_coor)
                distances, indices = nbrs.radius_neighbors(spatial_graph_coor, return_distance=True)
            elif spatial_graph_type == "knn":
                nbrs = NearestNeighbors(n_neighbors=spatial_graph_knn_cutoff).fit(spatial_graph_coor)
                distances, indices = nbrs.kneighbors(spatial_graph_coor, return_distance=True)
            else:
                raise NotImplementedError
            spatial_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
            spatial_graph_KNN_df = pd.concat(spatial_graph_KNN_list)
            spatial_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
            Spatial_Net = spatial_graph_KNN_df.copy()
            Spatial_Net = Spatial_Net[Spatial_Net['Distance']>0]
            Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
            Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
            print(f'The spatial graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
            print(f'{Spatial_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
            adata.uns['Spatial_Net'] = Spatial_Net
            spatial_graph_edge_index = []
            for idx, row in Spatial_Net.iterrows():
                cell1 = cell_to_index[row['Cell1']]
                cell2 = cell_to_index[row['Cell2']]
                spatial_graph_edge_index.append([cell1, cell2])
                spatial_graph_edge_index.append([cell2, cell1])  # Add reversed edge since the graph is undirected
            spatial_graph_edge_index = torch.tensor(spatial_graph_edge_index, dtype=torch.long).t().contiguous()
            
        # For gene graph
        if use_gene_graph:
            if gene_graph_pca_select:
                # Apply PCA to reduce dimensions to 50
                pca = PCA(n_components=gene_graph_pca_num)
                pca_data = pca.fit_transform(np.array(X_masked))
                nbrs_gene = NearestNeighbors(n_neighbors=gene_graph_knn_cutoff).fit(pca_data)
                distances, indices = nbrs_gene.kneighbors(pca_data, return_distance=True)
            else:
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=gene_graph_num_high_var_genes)
                hvg_indices = adata.var['highly_variable']
                hvg_data = torch.from_numpy(adata.X.astype(np.float32))[:, hvg_indices]
                nbrs_gene = NearestNeighbors(n_neighbors=gene_graph_knn_cutoff).fit(hvg_data)
                distances, indices = nbrs_gene.kneighbors(hvg_data, return_distance=True)
            gene_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
            gene_graph_KNN_df = pd.concat(gene_graph_KNN_list)
            gene_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
            Gene_Net = gene_graph_KNN_df.copy()
            Gene_Net['Cell1'] = Gene_Net['Cell1'].map(id_cell_trans)
            Gene_Net['Cell2'] = Gene_Net['Cell2'].map(id_cell_trans)
            print(f'The gene graph contains {Gene_Net.shape[0]} edges, {adata.n_obs} cells.')
            print(f'{Gene_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
            gene_graph_edge_index = []
            for idx, row in Gene_Net.iterrows():
                cell1 = cell_to_index[row['Cell1']]
                cell2 = cell_to_index[row['Cell2']]
                gene_graph_edge_index.append([cell1, cell2])
                gene_graph_edge_index.append([cell2, cell1])  # Add reversed edge since the graph is undirected
            gene_graph_edge_index = torch.tensor(gene_graph_edge_index, dtype=torch.long).t().contiguous()

        # **构建形态学图**
        if use_morphology_graph:
            image_embedding_path = f"{data_dir}/{sample_number}/extract_image_feature.npy"
            morphology_features = np.load(image_embedding_path)
            nbrs_morph = NearestNeighbors(n_neighbors=morphology_knn_cutoff).fit(morphology_features)
            distances, indices = nbrs_morph.kneighbors(morphology_features, return_distance=True)
            morphology_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
            morphology_graph_KNN_df = pd.concat(morphology_graph_KNN_list)
            morphology_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
            morphology_Net = morphology_graph_KNN_df.copy()
            morphology_Net['Cell1'] = morphology_Net['Cell1'].map(id_cell_trans)
            morphology_Net['Cell2'] = morphology_Net['Cell2'].map(id_cell_trans)
            print(f'The gene graph contains {morphology_Net.shape[0]} edges, {adata.n_obs} cells.')
            print(f'{morphology_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
            morphology_graph_edge_index = []
            for idx, row in morphology_Net.iterrows():
                cell1 = cell_to_index[row['Cell1']]
                cell2 = cell_to_index[row['Cell2']]
                morphology_graph_edge_index.append([cell1, cell2])
                morphology_graph_edge_index.append([cell2, cell1])  # Add reversed edge since the graph is undirected
            morphology_graph_edge_index = torch.tensor(morphology_graph_edge_index, dtype=torch.long).t().contiguous()
            print(f'Morphology graph contains {morphology_graph_edge_index.shape[1]} edges.')

        # **创建 PyG 数据对象**
        edge_index = torch.cat([spatial_graph_edge_index, gene_graph_edge_index, morphology_graph_edge_index], dim=1)
        edge_type = torch.cat([torch.zeros(spatial_graph_edge_index.size(1), dtype=torch.long),
                               torch.ones(gene_graph_edge_index.size(1), dtype=torch.long),
                               torch.full((morphology_graph_edge_index.size(1),), 2, dtype=torch.long)], dim=0)

        data = Data(x=torch.tensor(X_masked), edge_index=edge_index, edge_type=edge_type)
        return data, val_mask, test_mask, torch.tensor(X_masked), torch.tensor(X_dense)
