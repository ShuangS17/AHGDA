import sys
import gc
import random

import dgl
import dgl.function as fn

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score

sys.path.append('../data')
from data_loader import data_loader

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluator(gt, pred):
    gt = gt.cpu().squeeze()
    pred = pred.cpu().squeeze()
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')


def hg_propagate_feat_dgl(g, tgt_type, num_hops, max_length, echo=False):
    for hop in range(1, max_length):
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            # if hop == args.num_hops and dtype != tgt_type: continue
            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type) \
                      or (hop > num_hops):
                        continue
                    # if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                g.nodes[ntype].data.pop(k)
            # if echo and len(removes): print('remove', removes)
        gc.collect()
    return g


def hg_propagate_sparse_pyg(adjs, tgt_types, num_hops, max_length, prop_feats=False, echo=False, prop_device='cpu'):
    store_device = 'cpu'
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]

    label_feats = {k: v.clone() for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}

    for hop in range(2, max_length):
        new_adjs = {}
        for rtype_r, adj_r in label_feats.items():
            metapath_types = list(rtype_r)
            if len(metapath_types) == hop:
                dtype_r, stype_r = metapath_types[0], metapath_types[-1]
                for rtype_l, adj_l in adjs_g.items():
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:
                        name = f'{dtype_l}{rtype_r}'
                        if (hop == num_hops and dtype_l not in tgt_types) or (hop > num_hops):
                            continue
                        if name not in new_adjs:
                            # if echo: print('Generating ...', name)
                            if prop_device == 'cpu':
                                new_adjs[name] = adj_l.matmul(adj_r)
                            else:
                                with torch.no_grad():
                                    new_adjs[name] = adj_l.matmul(adj_r.to(prop_device)).to(store_device)
                        else:
                            if echo: print(f'Warning: {name} already exists')
        label_feats.update(new_adjs)

        removes = []
        for k in label_feats.keys():
            metapath_types = list(k)
            if metapath_types[0] in tgt_types: continue  # metapath should end with target type in label propagation
            if len(metapath_types) <= hop:
                removes.append(k)
        for k in removes:
            label_feats.pop(k)
        # if echo and len(removes): print('remove', removes)
        del new_adjs
        gc.collect()

    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()

    return label_feats


def check_acc(preds_dict, condition, init_labelS, train_nid, show_test=True, loss_type='ce'):
    mask_train = []
    remove_label_keys = []
    k = list(preds_dict.keys())[0]
    v = preds_dict[k]
    if loss_type == 'ce':
        na = len(train_nid)
    elif loss_type == 'bce':
        na = len(train_nid) * v.size(1)

    for k, v in preds_dict.items():
        if loss_type == 'ce':
            pred = v.argmax(1)
        elif loss_type == 'bce':
            pred = (v > 0).int()

        a = pred[train_nid] == init_labelS[train_nid]
        ra = a.sum() / na

        if loss_type == 'ce':
            vv = torch.log(v / (v.sum(1, keepdim=True) + 1e-6) + 1e-6)
            la = F.nll_loss(vv[train_nid], init_labelS[train_nid])
        else:
            vv = (v / 2. + 0.5).clamp(1e-6, 1-1e-6)
            la = F.binary_cross_entropy(vv[train_nid], init_labelS[train_nid].float())
        if condition(ra, k):
            mask_train.append(a)
        else:
            remove_label_keys.append(k)
        if show_test:
            print(f"{k}  acc={ra:.4f}  loss={la.item():.4f}")
        else:
            print(f"{k}  acc={ra:.4f}")
    print(set(list(preds_dict.keys())) - set(remove_label_keys))

    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / na)


def load_dataset(args):
    source_dl = data_loader(f'{args.root}/{args.datasetS}')
    target_dl = data_loader(f'{args.root}/{args.datasetT}')

    features_S = []
    for i in range(len(source_dl.nodes['count'])):
        th = source_dl.nodes['attr'][i]
        if th is None:
            features_S.append(torch.eye(source_dl.nodes['count'][i]))
        else:
            features_S.append(torch.FloatTensor(th))

    idx_shiftS = np.zeros(len(source_dl.nodes['count'])+1, dtype=np.int32)
    for i in range(len(source_dl.nodes['count'])):
        idx_shiftS[i+1] = idx_shiftS[i] + source_dl.nodes['count'][i]

    features_T = []
    for i in range(len(target_dl.nodes['count'])):
        th = target_dl.nodes['attr'][i]
        if th is None:
            features_T.append(torch.eye(target_dl.nodes['count'][i]))
        else:
            features_T.append(torch.FloatTensor(th))

    idx_shiftT = np.zeros(len(target_dl.nodes['count']) + 1, dtype=np.int32)
    for i in range(len(target_dl.nodes['count'])):
        idx_shiftT[i + 1] = idx_shiftT[i] + target_dl.nodes['count'][i]


    num_classes = source_dl.labels['num_classes']
    init_labelS = np.zeros((source_dl.nodes['count'][0], num_classes), dtype=int)
    init_labelT = np.zeros((target_dl.nodes['count'][0], num_classes), dtype=int)

    nidS = np.nonzero(source_dl.labels['mask'])[0]
    nidT = np.nonzero(target_dl.labels['mask'])[0]

    init_labelS[nidS] = source_dl.labels['data'][nidS]
    init_labelT[nidT] = target_dl.labels['data'][nidT]

    init_labelS = init_labelS.argmax(axis=1)
    init_labelS = torch.LongTensor(init_labelS)
    init_labelT = init_labelT.argmax(axis=1)
    init_labelT = torch.LongTensor(init_labelT)



    adjS = []
    for i, (k, v) in enumerate(source_dl.links['data'].items()):
        v = v.tocoo()
        col_type = np.where(idx_shiftS > v.col[0])[0][0] - 1
        row_type = np.where(idx_shiftS > v.row[0])[0][0] - 1
        row = v.row - idx_shiftS[row_type]
        col = v.col - idx_shiftS[col_type]
        sparse_sizes = (source_dl.nodes['count'][row_type], source_dl.nodes['count'][col_type])
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        adjS.append(adj)

    adjT = []
    for i, (k, v) in enumerate(target_dl.links['data'].items()):
        v = v.tocoo()
        col_type = np.where(idx_shiftT > v.col[0])[0][0] - 1
        row_type = np.where(idx_shiftT > v.row[0])[0][0] - 1
        row = v.row - idx_shiftT[row_type]
        col = v.col - idx_shiftT[col_type]
        sparse_sizes = (target_dl.nodes['count'][row_type], target_dl.nodes['count'][col_type])
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        adjT.append(adj)

    if args.datasetS in {'dblp11', 'dblp12', 'dblp13'}:
        P_s, A_s, V_s = features_S
        PP_s, PA_s, AP_s, PV_s, VP_s = adjS

        assert torch.all(PA_s.storage.col() == AP_s.t().storage.col())
        assert torch.all(PV_s.storage.col() == VP_s.t().storage.col())

        new_edges_S = {}
        ntypes_S = set()
        etypes_S  = [ # src->tgt
            ('P', 'P-P', 'P'),
            ('P', 'P-A', 'A'),
            ('A', 'A-P', 'P'),
            ('P', 'P-V', 'V'),
            ('V', 'V-P', 'P'),
        ]
        for etype, adj in zip(etypes_S , adjS):
            stype, rtype, dtype = etype
            src, dst, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges_S[(stype, rtype, dtype)] = (src, dst)
            ntypes_S.add(stype)
            ntypes_S.add(dtype)

        gS = dgl.heterograph(new_edges_S)

        gS.nodes['P'].data['P'] = P_s
        gS.nodes['A'].data['A'] = A_s
        gS.nodes['V'].data['V'] = V_s
    elif args.datasetS in {'IMDB1', 'IMDB2'}:
        M_s, A_s, D_s = features_S
        MA_s, AM_s, MD_s, DM_s = adjS

        new_edges_S = {}
        ntypes_S = set()
        etypes_S = [  # src->tgt
            ('M', 'M-A', 'A'),
            ('A', 'A-M', 'M'),
            ('M', 'M-D', 'D'),
            ('D', 'D-M', 'M'),
        ]
        for etype, adj in zip(etypes_S, adjS):
            stype, rtype, dtype = etype
            src, dst, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges_S[(stype, rtype, dtype)] = (src, dst)
            ntypes_S.add(stype)
            ntypes_S.add(dtype)
        gS = dgl.heterograph(new_edges_S)
        gS.nodes['M'].data['M'] = M_s
        gS.nodes['A'].data['A'] = A_s
        gS.nodes['D'].data['D'] = D_s
    else:
        assert 0

    if args.datasetT in {'dblp11', 'dblp12', 'dblp13'}:
        P_t, A_t, V_t = features_T
        PP_t, PA_t, AP_t, PV_t, VP_t = adjT

        assert torch.all(PA_t.storage.col() == AP_t.t().storage.col())
        assert torch.all(PV_t.storage.col() == VP_t.t().storage.col())

        new_edges_T = {}
        ntypes_T = set()
        etypes_T = [ # src->tgt
            ('P', 'P-P', 'P'),
            ('P', 'P-A', 'A'),
            ('A', 'A-P', 'P'),
            ('P', 'P-V', 'V'),
            ('V', 'V-P', 'P'),
        ]
        for etype, adj in zip(etypes_T, adjT):
            stype, rtype, dtype = etype
            src, dst, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges_T[(stype, rtype, dtype)] = (src, dst)
            ntypes_T.add(stype)
            ntypes_T.add(dtype)

        gT = dgl.heterograph(new_edges_T)
        gT.nodes['P'].data['P'] = P_t
        gT.nodes['A'].data['A'] = A_t
        gT.nodes['V'].data['V'] = V_t
    elif args.datasetT in {'IMDB1', 'IMDB2'}:
        M_t, A_t, D_t = features_T
        MA_t, AM_t, MD_t, DM_t = adjT

        new_edges_T = {}
        ntypes_T = set()
        etypes_T = [  # src->tgt
            ('M', 'M-A', 'A'),
            ('A', 'A-M', 'M'),
            ('M', 'M-D', 'D'),
            ('D', 'D-M', 'M'),
        ]
        for etype, adj in zip(etypes_T, adjT):
            stype, rtype, dtype = etype
            src, dst, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges_T[(stype, rtype, dtype)] = (src, dst)
            ntypes_T.add(stype)
            ntypes_T.add(dtype)
        gT = dgl.heterograph(new_edges_T)
        gT.nodes['M'].data['M'] = M_t
        gT.nodes['A'].data['A'] = A_t
        gT.nodes['D'].data['D'] = D_t
    else:
        assert 0

    if args.datasetS in {'IMDB1', 'IMDB2'}:
        adjS = {'MA': MA_s, 'AM': AM_s, 'MD': MD_s, 'DM': DM_s}
    elif args.datasetS in {'dblp11', 'dblp12', 'dblp13'}:
        adjS = {'PP': PP_s, 'PA': PA_s, 'AP': AP_s, 'PV': PV_s, 'VP': VP_s}
    else:
        assert 0

    if args.datasetT in {'IMDB1', 'IMDB2'}:
        adjT = {'MA': MA_t, 'AM': AM_t, 'MD': MD_t, 'DM': DM_t}
    elif args.datasetT in {'dblp11', 'dblp12', 'dblp13'}:
        adjT = {'PP': PP_t, 'PA': PA_t, 'AP': AP_t, 'PV': PV_t, 'VP': VP_t}
    else:
        assert 0

    return gS, gT, adjS, adjT, init_labelS, init_labelT, num_classes, source_dl, target_dl, nidS, nidT, features_S, features_T


def build_global_graph(dl):
    total_nodes = dl.nodes['total']

    node_types = np.zeros(total_nodes, dtype=int)
    num_types = len(dl.nodes['count'])

    for t_id in range(num_types):
        start = dl.nodes['shift'][t_id]
        count = dl.nodes['count'][t_id]
        end = start + count
        node_types[start:end] = t_id

    rows = []
    cols = []

    for r_id, adj in dl.links['data'].items():
        adj_coo = adj.tocoo()
        rows.append(adj_coo.row)
        cols.append(adj_coo.col)

        rows.append(adj_coo.col)
        cols.append(adj_coo.row)

    if not rows:
        return sp.csr_matrix((total_nodes, total_nodes)), node_types

    all_row = np.concatenate(rows)
    all_col = np.concatenate(cols)
    data = np.ones_like(all_row)

    global_adj = sp.csr_matrix((data, (all_row, all_col)), shape=(total_nodes, total_nodes))
    global_adj.data = np.ones_like(global_adj.data)

    return global_adj, node_types