import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
import networkx as nx

def calculate_weight(homo_adj, global_adj, node_types, target_idx_range):
    target_start, target_end = target_idx_range
    num_p_nodes = target_end - target_start

    if not sp.isspmatrix_csr(homo_adj):
        homo_adj = homo_adj.tocsr()

    G = nx.from_scipy_sparse_array(homo_adj)

    # Eigenvector Centrality
    eigen_dict = nx.eigenvector_centrality_numpy(G)
    eigen = np.array([eigen_dict.get(i, 0) for i in range(num_p_nodes)])

    # Clustering Coefficient
    cluster_dict = nx.clustering(G)
    cluster = np.array([cluster_dict.get(i, 0) for i in range(num_p_nodes)])

    # (m_i / V_T)
    unique_types = np.unique(node_types)
    V_T = len(unique_types)
    vals = np.zeros(num_p_nodes)

    adj_p_rows = global_adj[target_start:target_end, :]
    if not sp.isspmatrix_csr(adj_p_rows):
        adj_p_rows = adj_p_rows.tocsr()

    indices = adj_p_rows.indices
    indptr = adj_p_rows.indptr

    for i in range(num_p_nodes):
        neighber = indices[indptr[i]:indptr[i + 1]]
        if len(neighber) > 0:
            nbr_types = node_types[neighber]
            m_i = len(np.unique(nbr_types))
            vals[i] = m_i / V_T

    # print(f"Eigen (Top 20):   {np.round(np.sort(eigen)[::-1][0:20], 4)}")
    # print(f"Cluster (Top 20): {np.round(np.sort(cluster)[::-1][0:20], 4)}")
    # print(f"Vals (Top 20):    {np.round(np.sort(vals)[::-1][0:20], 4)}")

    scaler = MinMaxScaler()
    logits = eigen+cluster+vals
    weight = scaler.fit_transform(logits.reshape(-1, 1)).flatten()
    print(f"weight (Top 20): {np.round(np.sort(weight)[::-1][:20], 4)}")
    return weight


def calculate_R(embeddings, cluster_labels, centers):
    my_centers = centers[cluster_labels]
    dis = np.linalg.norm(embeddings - my_centers, axis=1)
    r_score = 1.0 / (1.0 + dis)
    return r_score

def calculate_U(outputs):
    probs = np.clip(outputs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log2(probs), axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    u_score = entropy - margin
    return u_score


def calculate_D(d_probs):
    d_val = np.clip(d_probs, 0.0, 0.9999)
    d_score = d_val / (1.0 - d_val)
    return d_score

def propagate_on_graph(adj, metric_val, weight, hop):
    num_nodes = adj.shape[0]
    assert metric_val.shape[0] == num_nodes, f"Metric size {metric_val.shape} mismatch with Adj size {num_nodes}"
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()

    adj = adj + sp.eye(num_nodes)

    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1.0)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv @ adj

    weighted_metric = weight * metric_val
    weighted_metric = norm_adj @ weighted_metric

    return weighted_metric

def active_select(embeddings, outputs, global_adj, pool_idx, budget, d_probs, node_types, class_num, target_idx_range):
    target_start, target_end = target_idx_range

    adj_pp = global_adj[target_start:target_end, target_start:target_end]
    all_indices = np.arange(global_adj.shape[0])
    non_p_mask = (all_indices < target_start) | (all_indices >= target_end)

    adj_p_all = global_adj[target_start:target_end, :]
    adj_p_other = adj_p_all[:, non_p_mask]
    adj_p_other_p = adj_p_other @ adj_p_other.T

    homo_adj = adj_pp + adj_p_other_p
    homo_adj = homo_adj.tocsr()
    # 权重二值化或者保留权重.这里保留权重
    # homo_adj.data = np.ones_like(homo_adj.data) # 二值化取消注释

    pseudo_labels = np.argmax(outputs, axis=1)
    emb_dim = embeddings.shape[1]
    centroids = np.zeros((class_num, emb_dim))

    for c in range(class_num):
        mask = (pseudo_labels == c)
        centroids[c] = np.mean(embeddings[mask], axis=0)
    cluster_labels = pseudo_labels

    weight = calculate_weight(homo_adj, global_adj, node_types, target_idx_range)
    # weight = np.ones(num_nodes, dtype=np.float32)

    R = calculate_R(embeddings, cluster_labels, centroids)
    U = calculate_U(outputs)
    D = calculate_D(d_probs)

    R_final = propagate_on_graph(homo_adj, R, weight, hop=1)
    U_final = propagate_on_graph(homo_adj, U, weight, hop=1)
    D_final = propagate_on_graph(homo_adj, D, weight, hop=1)

    scaler = MinMaxScaler()
    norm_R = scaler.fit_transform(R_final.reshape(-1, 1)).flatten()
    norm_U = scaler.fit_transform(U_final.reshape(-1, 1)).flatten()
    norm_D = scaler.fit_transform(D_final.reshape(-1, 1)).flatten()

    a=0.2
    b=0.3

    score_total = a * norm_R + b * norm_U + (1 - a - b) * norm_D

    selected_indices = []

    quota = budget // class_num
    remain = budget % class_num

    cluster_pool = {c: [] for c in range(class_num)}
    for idx in pool_idx:
        local_idx = idx - target_start
        if 0 <= local_idx < len(cluster_labels):
            c_lab = cluster_labels[local_idx]
            cluster_pool[c_lab].append(idx)
        else:
            print(f"Warning: Index {idx} out of target range {target_idx_range}")

    quotas = {c: quota for c in range(class_num)}
    cluster_sizes = sorted([(c, len(idxs)) for c, idxs in cluster_pool.items()], key=lambda x: x[1], reverse=True)

    for i in range(remain):
        quotas[cluster_sizes[i][0]] += 1

    for c in range(class_num):
        quota = quotas[c]
        candidates = cluster_pool[c]
        if not candidates or quota == 0:
            continue

        cand_local_indices = [idx - target_start for idx in candidates]
        cand_scores_val = score_total[cand_local_indices]

        sorted_local_args = np.argsort(-cand_scores_val)

        for i in range(min(quota, len(candidates))):
            original_idx = candidates[sorted_local_args[i]]
            selected_indices.append(original_idx)

    if len(selected_indices) < budget:
        needed = budget - len(selected_indices)
        print('remain need', needed)
        remaining = list(set(pool_idx) - set(selected_indices))
        if remaining:
            rem_local = [idx - target_start for idx in remaining]
            rem_scores = score_total[rem_local]
            rem_args = np.argsort(-rem_scores)[:needed]
            for arg in rem_args:
                selected_indices.append(remaining[arg])

    return selected_indices