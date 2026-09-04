import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag

from model import *
from utils import *
from sparse_tools import SparseAdjList
from flip_gradient import GradReverse
from Active_select import active_select

def main(args):
    PSEUDO_LABEL_THRESHOLD = args.pseudo_threshold
    (gS, gT, adjS, adjT, init_labelS, init_labelT, num_classes,
     source_dl, target_dl, nidS, nidT, features_S, features_T) = load_dataset(args)

    prop_device = 'cpu'
    device = f'cuda:{args.gpu}' if not args.cpu else 'cpu'

    for k in adjS.keys():
        adjS[k].storage._value = None
        adjS[k].storage._value = torch.ones(adjS[k].nnz()) / adjS[k].sum(dim=-1)[adjS[k].storage.row()]
    for k in adjT.keys():
        adjT[k].storage._value = None
        adjT[k].storage._value = torch.ones(adjT[k].nnz()) / adjT[k].sum(dim=-1)[adjT[k].storage.row()]

    if args.datasetS in {'dblp11', 'dblp12', 'dblp13'}:
        tgt_type = 'P'
        node_types = ['P', 'A', 'V']
    elif args.datasetS in {'IMDB1', 'IMDB2'}:
        tgt_type = 'M'
        node_types = ['M', 'A', 'D']
    else:
        assert 0

    # features_S: 0:P/M, 1:A, 2:V/D
    raw_A_S = features_S[1].to(device)
    raw_A_T = features_T[1].to(device)
    raw_V_S = features_S[2].to(device)
    raw_V_T = features_T[2].to(device)

    neighbor_types = [t for t in node_types if t != tgt_type]

    # compute k-hop features
    max_length = args.num_hops + 1

    gS = hg_propagate_feat_dgl(gS, tgt_type, args.num_hops, max_length, echo=True)
    gT = hg_propagate_feat_dgl(gT, tgt_type, args.num_hops, max_length, echo=True)
    raw_featS = {}
    raw_featT = {}
    keyS = list(gS.nodes[tgt_type].data.keys())
    keyT = list(gT.nodes[tgt_type].data.keys())

    for k in keyS:
        raw_featS[k] = gS.nodes[tgt_type].data.pop(k)
    for k in keyT:
        raw_featT[k] = gT.nodes[tgt_type].data.pop(k)

    all_keys = sorted(set(raw_featS.keys()) | set(raw_featT.keys()))
    data_size = {k: raw_featS.get(k, raw_featT.get(k)).size(-1) for k in all_keys}
    for ntype in neighbor_types:
        if ntype not in data_size:
            data_size[ntype] = features_S[node_types.index(ntype)].shape[1]
    gc.collect()

    labelS = init_labelS.clone()
    labelT = init_labelT.clone()

    for seed in args.seeds:
        args.seed = seed
        set_random_seed(args.seed)

        num_nodeS = source_dl.nodes['count'][0]
        featS = {k: v.detach().clone() for k, v in raw_featS.items()}
        featT = {k: v.detach().clone() for k, v in raw_featT.items()}

        # labels propagate alongside the metapath
        label_feats_S = {}
        if args.label_feats:
            label_onehot = torch.zeros((num_nodeS, num_classes))
            label_onehot[nidS] = F.one_hot(init_labelS[nidS], num_classes).float()
            max_length = args.num_label_hops + 1

            # compute k-hop feature
            meta_adjS = hg_propagate_sparse_pyg(
                adjS, tgt_type, args.num_label_hops, max_length,
                prop_feats=False, echo=True, prop_device=prop_device)
            meta_adjT = hg_propagate_sparse_pyg(
                adjT, tgt_type, args.num_label_hops, max_length,
                prop_feats=False, echo=False, prop_device=prop_device)

            for k, v in tqdm(meta_adjS.items()):
                label_feats_S[k] = remove_diag(v) @ label_onehot
            gc.collect()

            condition = lambda ra, k: True
            check_acc(label_feats_S, condition, init_labelS, nidS, show_test=True)

        if not args.cpu: torch.cuda.empty_cache()
        gc.collect()

        source_batch_size = args.batch_size_P // 2
        target_batch_size = args.batch_size_P - source_batch_size

        loader_P_S = torch.utils.data.DataLoader(
            nidS, batch_size=source_batch_size, shuffle=True, drop_last=True)
        loader_P_T = torch.utils.data.DataLoader(
            nidT, batch_size=target_batch_size, shuffle=True, drop_last=True)

        ids_A_S = torch.arange(raw_A_S.size(0))
        ids_A_T = torch.arange(raw_A_T.size(0))
        loader_A_S = torch.utils.data.DataLoader(ids_A_S, batch_size=args.batch_size_A, shuffle=True, drop_last=True)
        loader_A_T = torch.utils.data.DataLoader(ids_A_T, batch_size=args.batch_size_A, shuffle=True, drop_last=True)

        ids_V_S = torch.arange(raw_V_S.size(0))
        ids_V_T = torch.arange(raw_V_T.size(0))
        loader_V_S = torch.utils.data.DataLoader(ids_V_S, batch_size=args.batch_size_V, shuffle=True, drop_last=True)
        loader_V_T = torch.utils.data.DataLoader(ids_V_T, batch_size=args.batch_size_V, shuffle=True, drop_last=True)

        iter_A_S = itertools.cycle(loader_A_S)
        iter_A_T = itertools.cycle(loader_A_T)
        iter_V_S = itertools.cycle(loader_V_S)
        iter_V_T = itertools.cycle(loader_V_T)

        domain_labels_A_batch = torch.cat([
            torch.zeros(args.batch_size_A, device=device, dtype=torch.long),
            torch.ones(args.batch_size_A, device=device, dtype=torch.long)
        ])
        domain_labels_V_batch = torch.cat([
            torch.zeros(args.batch_size_V, device=device, dtype=torch.long),
            torch.ones(args.batch_size_V, device=device, dtype=torch.long)
        ])

        model = SeHGNN(args.hidden1, args.hidden2, num_classes, all_keys, label_feats_S.keys(), tgt_type,
                       args.dropout, args.input_drop, args.att_drop, args.n_fp_layers, args.n_task_layers, args.act,
                       args.residual, data_size=data_size)
        model = model.to(device)

        loss_fcn = nn.CrossEntropyLoss()
        ad_loss_f = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

        best_epoch = -1
        best_test_acc = (0,0)
        label_feats_T = {}
        cached_T_label_onehot = None

        domain_label_S = torch.zeros(source_batch_size, device=device, dtype=torch.long)
        domain_label_T = torch.ones(target_batch_size, device=device, dtype=torch.long)
        domain_labels = torch.cat((domain_label_S, domain_label_T), dim=0).to(device)

        num_meta = model.num_channels
        domain_label_meta_S = torch.zeros(source_batch_size * num_meta, device=device, dtype=torch.long)
        domain_label_meta_T = torch.ones(target_batch_size * num_meta, device=device, dtype=torch.long)
        domain_labels_meta = torch.cat((domain_label_meta_S, domain_label_meta_T), dim=0)

        print(" ====== Phase 1: Initial Training ======")
        for epoch in tqdm(range(args.epoch)):
            gc.collect()
            if not args.cpu: torch.cuda.synchronize()

            model.train()
            p = epoch / args.epoch
#             grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1
            grl_lambda = p
            lambda_max = 0.1
            if grl_lambda > lambda_max:
                grl_lambda = lambda_max
            GradReverse.rate = grl_lambda

            total_cls_loss = 0.0
            total_ad_loss_P = 0.0
            total_ad_loss_A = 0.0
            total_ad_loss_V = 0.0
            total_ad_loss_path = 0.0
            total_loss = 0.0
            num_batches = 0

            epoch_iterator = tqdm(zip(loader_P_S, loader_P_T, iter_A_S, iter_A_T, iter_V_S, iter_V_T), desc=f"Epoch {epoch}", leave=False)
            for batch_nid_S, batch_nid_T, b_id_A_S, b_id_A_T, b_id_V_S, b_id_V_T in epoch_iterator:
                batch_featS = {k: v[batch_nid_S].to(device) for k, v in featS.items()}
                batch_label_featsS = {k: v[batch_nid_S].to(device) for k, v in label_feats_S.items()}
                batch_featT = {k: v[batch_nid_T].to(device) for k, v in featT.items()}
                batch_label_featsT = {k: v[batch_nid_T].to(device) for k, v in label_feats_T.items()} if label_feats_T else None

                optimizer.zero_grad()
                output_S, output_T, d_logits_P, d_logits_meta, _, _ = model(batch_featS, batch_label_featsS, batch_featT, batch_label_featsT)

                feat_A_S_batch = raw_A_S[b_id_A_S]
                feat_A_T_batch = raw_A_T[b_id_A_T]
                d_logits_A = model.forward_neighbor_alignment('A', feat_A_S_batch, feat_A_T_batch)
                ad_loss_A = ad_loss_f(d_logits_A, domain_labels_A_batch)

                feat_V_S_batch = raw_V_S[b_id_V_S]
                feat_V_T_batch = raw_V_T[b_id_V_T]
                d_logits_V = model.forward_neighbor_alignment(neighbor_types[1], feat_V_S_batch, feat_V_T_batch)
                ad_loss_V = ad_loss_f(d_logits_V, domain_labels_V_batch)

                cls_loss = loss_fcn(output_S, labelS[batch_nid_S].long().to(device))
                ad_loss_P = ad_loss_f(d_logits_P, domain_labels)

                ad_loss_path = ad_loss_f(d_logits_meta, domain_labels_meta)
                total_loss_batch = cls_loss + args.ad_weightnode*(ad_loss_P+ad_loss_A+ad_loss_V) + args.ad_weightpath*ad_loss_path

                total_loss_batch.backward()
                optimizer.step()

                total_cls_loss += cls_loss.item()
                total_ad_loss_P += ad_loss_P.item()
                total_ad_loss_A += ad_loss_A.item()
                total_ad_loss_V += ad_loss_V.item()
                total_ad_loss_path += ad_loss_path.item()
                total_loss += total_loss_batch.item()
                num_batches += 1

                epoch_iterator.set_postfix({
                    'cls_loss': f'{cls_loss.item():.4f}',
                    'ad_loss': f'{ad_loss_P.item():.4f}'
                })

            # Generate pseudo-labels
            with torch.no_grad():
                model.eval()
                featT_device = {k: v.to(device) for k, v in featT.items()}
                label_featT_device = {k: v.to(device) for k, v in
                                      label_feats_T.items()} if label_feats_T else None
                _, output_T_full, _, _, _, _ = model(None, None, featT_device, label_featT_device)

                probs_T = F.softmax(output_T_full, dim=1)
                detached_probs_T = probs_T.detach()
                max_probs, pseudo_labels = torch.max(detached_probs_T, dim=1)

                mask = max_probs > PSEUDO_LABEL_THRESHOLD
                confident_labels = pseudo_labels[mask].cpu()
                pseudo_label_onehot = torch.zeros((len(nidT), num_classes))
                confident_indices = mask.cpu().nonzero(as_tuple=True)[0]
                pseudo_label_onehot[confident_indices] = F.one_hot(confident_labels, num_classes).float()
                cached_T_label_onehot = pseudo_label_onehot

                for k, v in meta_adjT.items():
                    label_feats_T[k] = remove_diag(v) @ pseudo_label_onehot
                    label_feats_T[k] = label_feats_T[k].to(device)

                # Evaluate
                featS_device = {k: v.to(device) for k, v in featS.items()}
                label_featS_device = {k: v.to(device) for k, v in label_feats_S.items()}
                label_featT_device = {k: v.to(device) for k, v in label_feats_T.items()} if label_feats_T else None

                output_S_eval, output_T_eval, d_logits_eval, _, _, emb_T_eval = model(featS_device, label_featS_device, featT_device, label_featT_device)

                pred_S = output_S_eval.cpu().argmax(dim=-1)
                train_acc = evaluator(labelS[nidS], pred_S[nidS])
                pred_T = output_T_eval.cpu().argmax(dim=-1)
                test_acc = evaluator(labelT[nidT], pred_T[nidT])

                log = (
                    f'\nEpoch {epoch:03d}   '
                    f'cls_loss: {total_cls_loss / num_batches:.4f}, ad_loss_P: {total_ad_loss_P / num_batches:.4f}, '
                    f'ad_loss_A: {total_ad_loss_A / num_batches:.4f}, ad_loss_V: {total_ad_loss_V / num_batches:.4f},'
                    f'ad_loss_path: {total_ad_loss_path / num_batches:.4f}, '
                    f'total_loss: {total_loss / num_batches:.4f}\n')
                log += f'Train Acc: micro {train_acc[0] * 100:.2f}%, macro {train_acc[1] * 100:.2f}%\n'
                log += f'Test Acc: micro {test_acc[0] * 100:.2f}%, macro {test_acc[1] * 100:.2f}%\n'

                if test_acc[0] > best_test_acc[0]:
                    best_test_acc = test_acc
                    best_epoch = epoch
            if not args.cpu: torch.cuda.empty_cache()
            tqdm.write(f"{log}")
        print(
            f'Best Epoch: {best_epoch}, Best Target Acc: micro {best_test_acc[0] * 100:.2f}%, macro {best_test_acc[1] * 100:.2f}%')
        print('\n')
        print(" ====== Phase 2: Active Learning ======")
        active_node_label = {}  # Store the real labels of actively selected nodes {node_id: label}

        d_probs_eval = F.softmax(d_logits_eval, dim=1)[:, 1].detach().cpu().numpy()
        d_probs = d_probs_eval[num_nodeS:]
        output_T = output_T_eval.detach().cpu()
        embeddings_T = output_T.numpy()
        prediction_T = F.softmax(output_T, dim=1).numpy()

        global_adjT, global_node_types = build_global_graph(target_dl)

        tgt_start = target_dl.nodes['shift'][0]
        tgt_count = target_dl.nodes['count'][0]
        tgt_end = tgt_start + tgt_count
        target_idx_range = (tgt_start, tgt_end)

        selected_indices= active_select(
            embeddings_T, prediction_T, global_adjT, list(nidT), args.active_budget, d_probs, global_node_types, num_classes, target_idx_range)

        print(f"Selected Node IDs: {sorted(selected_indices)}")

        for idx in selected_indices:
            if idx not in active_node_label:
                active_node_label[idx] = labelT[idx].item()
        print(f"Selected {len(selected_indices)} nodes.")

        new_nidT = np.setdiff1d(nidT, selected_indices)
        print(f"new target dataset size: {len(new_nidT)}")

        print('\n')

        iter_A_S = itertools.cycle(loader_A_S)
        iter_A_T = itertools.cycle(loader_A_T)
        iter_V_S = itertools.cycle(loader_V_S)
        iter_V_T = itertools.cycle(loader_V_T)

        print(" ====== Phase 3: Fine-Tuning ======")
        active_index = torch.LongTensor(list(active_node_label.keys()))
        active_label = torch.LongTensor(list(active_node_label.values()))

        best_acc_final = (0,0)
        best_ft_epoch = -1
        T_label_onehot_final = cached_T_label_onehot.clone()

        if len(active_index) > 0:
            T_label_onehot_final[active_index] = F.one_hot(active_label, num_classes).float()

        for k, v in meta_adjT.items():
            label_feats_T[k] = remove_diag(v) @ T_label_onehot_final
            label_feats_T[k] = label_feats_T[k].to(device)

        for ft_epoch in range(args.fine_tune_epochs):
            model.train()

            ft_cls_loss = 0.0
            ft_ad_loss_P = 0.0
            ft_ad_loss_A = 0.0
            ft_ad_loss_V = 0.0
            ft_ad_loss_path = 0.0
            ft_al_loss = 0.0
            total_ft_loss = 0.0
            num_batches = 0

            ft_epoch_iterator = tqdm(zip(loader_P_S, loader_P_T, iter_A_S, iter_A_T, iter_V_S, iter_V_T), desc=f"ft_epoch {ft_epoch}", leave=False)
            for batch_nid_S, batch_nid_T, b_id_A_S, b_id_A_T, b_id_V_S, b_id_V_T in ft_epoch_iterator:
                batch_featS = {k: v[batch_nid_S].to(device) for k, v in featS.items()}
                batch_label_featsS = {k: v[batch_nid_S].to(device) for k, v in label_feats_S.items()}
                batch_featT = {k: v[batch_nid_T].to(device) for k, v in featT.items()}
                batch_label_featsT = {k: v[batch_nid_T].to(device) for k, v in label_feats_T.items()} if label_feats_T else None

                optimizer.zero_grad()

                output_S, output_T, d_logits_P, d_logits_meta, _, _ = model(batch_featS, batch_label_featsS, batch_featT, batch_label_featsT)

                feat_A_S_batch = raw_A_S[b_id_A_S]
                feat_A_T_batch = raw_A_T[b_id_A_T]
                d_logits_A = model.forward_neighbor_alignment('A', feat_A_S_batch, feat_A_T_batch)
                ad_loss_A = ad_loss_f(d_logits_A, domain_labels_A_batch)

                feat_V_S_batch = raw_V_S[b_id_V_S]
                feat_V_T_batch = raw_V_T[b_id_V_T]
                d_logits_V = model.forward_neighbor_alignment(neighbor_types[1], feat_V_S_batch, feat_V_T_batch)
                ad_loss_V = ad_loss_f(d_logits_V, domain_labels_V_batch)

                cls_loss = loss_fcn(output_S, labelS[batch_nid_S].long().to(device))
                ad_loss_P = ad_loss_f(d_logits_P, domain_labels)
                ad_loss_path = ad_loss_f(d_logits_meta, domain_labels_meta)

                al_loss = 0.0
                batch_active_mask = (batch_nid_T.cpu().unsqueeze(1) == active_index.unsqueeze(0)).any(dim=1)
                if batch_active_mask.any():
                    pred_active = output_T[batch_active_mask.to(device)]
                    curr_nids = batch_nid_T.cpu()[batch_active_mask].numpy()
                    curr_labels = [active_node_label[nid] for nid in curr_nids]
                    al_loss = loss_fcn(pred_active, torch.LongTensor(curr_labels).to(device))

                loss_term_al = al_loss if isinstance(al_loss, torch.Tensor) else torch.tensor(0.0, device=device)
                ft_loss = cls_loss + args.ad_weightnode*(ad_loss_P+ad_loss_A+ad_loss_V) + args.ad_weightpath*ad_loss_path + args.al_weight * loss_term_al
                ft_loss.backward()
                optimizer.step()

                ft_cls_loss += cls_loss.item()
                ft_ad_loss_P += ad_loss_P.item()
                ft_ad_loss_A += ad_loss_A.item()
                ft_ad_loss_V += ad_loss_V.item()
                ft_ad_loss_path += ad_loss_path.item()
                ft_al_loss += al_loss
                total_ft_loss += ft_loss.item()
                num_batches += 1

            with torch.no_grad():
                model.eval()
                featS_device = {k: v.to(device) for k, v in featS.items()}
                label_featS_device = {k: v.to(device) for k, v in label_feats_S.items()}
                featT_device = {k: v.to(device) for k, v in featT.items()}
                label_featT_device = {k: v.to(device) for k, v in label_feats_T.items()} if label_feats_T else None

                output_S_eval, output_T_eval, _, _ ,_, _ = model(featS_device, label_featS_device, featT_device,
                                                        label_featT_device)

                pred_S = output_S_eval.cpu().argmax(dim=-1)
                train_acc = evaluator(labelS[nidS], pred_S[nidS])
                pred_T = output_T_eval.cpu().argmax(dim=-1)
                test_acc = evaluator(labelT[new_nidT], pred_T[new_nidT])

                log = (
                    f'\nft_epoch {ft_epoch:03d},'
                    f'ft_cls_loss: {ft_cls_loss / num_batches:.4f}, ft_ad_loss_P: {ft_ad_loss_P / num_batches:.4f},'
                    f'ft_ad_loss_A: {ft_ad_loss_A / num_batches:.4f}, ft_ad_loss_V: {ft_ad_loss_V / num_batches:.4f},'
                    f'ft_ad_loss_path: {ft_ad_loss_path / num_batches:.4f}, '
                    f'ft_al_loss: {ft_al_loss / num_batches:.4f}, total_ft_loss: {total_ft_loss / num_batches:.4f}\n')
                log += f'Train Acc: micro {train_acc[0] * 100:.2f}%, macro {train_acc[1] * 100:.2f}%\n'
                log += f'Test Acc: micro {test_acc[0] * 100:.2f}%, macro {test_acc[1] * 100:.2f}%\n'
                if test_acc[0] > best_acc_final[0]:
                    best_acc_final = test_acc
                    best_ft_epoch = ft_epoch
            if not args.cpu: torch.cuda.empty_cache()
            print(f"{log}")
        print(
                f'Best Epoch: {best_ft_epoch}, Best Target Acc: micro {best_acc_final[0] * 100:.2f}%, macro {best_acc_final[1] * 100:.2f}%')

        del model
        if not args.cpu: torch.cuda.empty_cache()


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='HGADA')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1],
                        help='the seed used in the training')
    parser.add_argument('--datasetS', type=str, default='dblp11',
                        choices=['dblp11','dblp12','dblp13', 'IMDB1', 'IMDB2'])
    parser.add_argument('--datasetT', type=str, default='dblp11',
                        choices=['dblp11','dblp12','dblp13', 'IMDB1', 'IMDB2'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--root', type=str, default='../data/')
    parser.add_argument('--epoch', type=int, default=50, help='Maxinum number of epochs.')
    parser.add_argument('--hidden1', type=int, default=256,
                        help='inital embedding size of nodes with no attributes')
    parser.add_argument('--num-hops', type=int, default=2,
                        help='number of hops for propagation of raw labels')
    parser.add_argument('--label-feats', action='store_true', default=False,
                        help='whether to use the label propagated features')
    parser.add_argument('--num-label-hops', type=int, default=2,
                        help='number of hops for propagation of raw features')
    ## For network structure
    parser.add_argument('--n-fp-layers', type=int, default=2,
                        help='the number of mlp layers for feature projection')
    parser.add_argument('--n-task-layers', type=int, default=3,
                        help='the number of mlp layers for the downstream task')
    parser.add_argument('--hidden2', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout on activation')
    parser.add_argument('--input-drop', type=float, default=0.1,
                        help='input dropout of input features')
    parser.add_argument('--att-drop', type=float, default=0.,
                        help='attention dropout of model')
    parser.add_argument('--act', type=str, default='none',
                        choices=['none', 'relu', 'leaky_relu', 'sigmoid'],
                        help='the activation function of the transformer part')
    parser.add_argument('--residual', action='store_true', default=False,
                        help='whether to add residual branch the raw input features')

    parser.add_argument('--amp', action='store_true', default=False,
                        help='whether to amp to accelerate training with float16(half) calculation')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--ad-weightnode', type=float, default=1)
    parser.add_argument('--ad-weightpath', type=float, default=1)
    parser.add_argument('--batch-size-P', type=int, default=10000)
    parser.add_argument('--batch-size-A', type=int, default=10000)
    parser.add_argument('--batch-size-V', type=int, default=10000)
    parser.add_argument('--pseudo-threshold', type=float, default=0.5)
    parser.add_argument('--active-budget', type=int, default=80, help='Number of nodes to select for labeling.')
    parser.add_argument('--fine-tune-epochs', type=int, default=20, help='Number of epochs for fine-tuning after AL.')
    parser.add_argument('--al-weight', type=float, default=0.01, help='Weight for al loss')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    # print(args)
    main(args)
