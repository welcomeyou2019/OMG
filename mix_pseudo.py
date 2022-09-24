# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
import os.path as osp
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from tu_dataset import TUDataset
# from torch_geometric.datasets import TUDataset
# from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
try:
    from torch_geometric.data.dataloader import Collater
except:
    from torch_geometric.loader.dataloader import Collater
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import faiss
import random

def query_gin_data(args, noise_idx=None):
    BATCH = args.batch_size
    if 'ogb' not in args.dataset:
        if 'IMDB' in args.dataset:  # IMDB-BINARY or #IMDB-MULTI
            class MyFilter(object):
                def __call__(self, data):
                    return data.num_nodes <= 70
                    # return True

            class MyPreTransform(object):
                def __call__(self, data):
                    data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                    data.x = F.one_hot(data.x, num_classes=136).to(torch.float)  # 136 in k-gnn?
                    return data

            path = osp.join(
                osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=MyPreTransform(),
                pre_filter=MyFilter(),
                noise_idx=noise_idx)

        elif args.dataset == 'PROTEINS':
            class MyFilter(object):
                def __call__(self, data):
                    # return not (data.num_nodes == 7 and data.num_edges == 12) and data.num_nodes < 450
                    return True

            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PROTEINS')
            dataset = TUDataset(path, name='PROTEINS', pre_filter=MyFilter(),noise_idx=noise_idx)

        elif 'REDDIT' in args.dataset:
            # raise ValueError
            class MyFilter(object):
                def __call__(self, data):
                    return data.num_nodes <= 3783
                    # return True

            class MyPreTransform(object):
                def __call__(self, data):
                    data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                    data.x = F.one_hot(data.x, num_classes=3782).to(torch.float)  # 136 in k-gnn?
                    return data

            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=MyPreTransform(),
                pre_filter=MyFilter(),noise_idx=noise_idx)

        elif 'COLLAB' in args.dataset:
            class MyFilter(object):
                def __call__(self, data):
                    return data.num_nodes <= 493
                    # return True

            class MyPreTransform(object):
                def __call__(self, data):
                    data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                    data.x = F.one_hot(data.x, num_classes=492).to(torch.float)  # 136 in k-gnn?
                    return data

            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=MyPreTransform(),
                pre_filter=MyFilter(),noise_idx=noise_idx)

        elif 'COIL-RAG' in args.dataset:
            class MyFilter(object):
                def __call__(self, data):
                    # return data.num_nodes <= 493
                    return True

            class MyPreTransform(object):
                def __call__(self, data):
                    data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                    data.x = F.one_hot(data.x, num_classes=11).to(torch.float)  # 136 in k-gnn?
                    return data

            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=MyPreTransform(),
                pre_filter=MyFilter(),noise_idx=noise_idx)

        else:
            class MyFilter(object):
                def __call__(self, data):
                    return True
            # print('here')
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
            dataset = TUDataset(path, name=args.dataset, pre_filter=MyFilter(),noise_idx=noise_idx)
    else:
        from ogb.graphproppred import PygGraphPropPredDataset
        from data import policy2transform
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
        # dataset = PygGraphPropPredDataset(root="dataset/" + 'node_deleted',
        #                                   name=args.dataset,
        #                                   pre_transform=policy2transform(policy='node_deleted', num_hops=2),
        #                                   )
        dataset = PygGraphPropPredDataset(name=args.dataset)

    args.feature_dim = dataset.data.x.shape[-1]
    return dataset

def _noise(x, add_noise_level=0.0, mult_noise_level=0.0, sparsity_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    # with torch.cuda.device(0):
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.FloatTensor(x.shape).normal_()
        #torch.clamp(add_noise, min=-(2*var), max=(2*var), out=add_noise) # clamp
    if mult_noise_level > 0.0:
        mult_noise = mult_noise_level * np.random.beta(2, 5) * (2*torch.FloatTensor(x.shape).uniform_()-1) + 1
    return mult_noise * x + add_noise

def do_noisy_mixup(x, y, alpha=0.0, add_noise_level=0.0, mult_noise_level=0.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0.0 else 1.0
    positive_idx = torch.zeros(y.shape).long()
    negative_idx = torch.zeros(y.shape).long()

    one_idx = torch.where(y==1)[0].numpy()
    zero_idx = torch.where(y==0)[0].numpy()
    # print(one_idx)
    positive_one_index = np.random.permutation(one_idx)
    positive_zero_idx = np.random.permutation(zero_idx)
    positive_idx[one_idx] = torch.from_numpy(positive_one_index).long()
    positive_idx[zero_idx] = torch.from_numpy(positive_zero_idx).long()
    # print(positive_idx)
    #positive pair
    positive_mix = lam * x + (1 - lam) * x[positive_idx]
    positive_pair_logic = torch.einsum('nc,nc->n', [x, x[positive_idx]]).unsqueeze(-1)

    negative_one_idx = np.random.choice(zero_idx, len(one_idx))
    negative_zero_idx = np.random.choice(one_idx, len(zero_idx))
    negative_idx[one_idx] = torch.from_numpy(negative_one_idx).long()
    negative_idx[zero_idx] = torch.from_numpy(negative_zero_idx).long()

    negative_mix = lam * x + (1-lam) * x[negative_idx]
    negative_pair_logic = torch.einsum('nc,nc->n', [x, x[negative_idx]]).unsqueeze(-1)

    labels = torch.zeros(2*x.shape[0], dtype=torch.long)
    labels[x.shape[0]:] = 1
    return positive_mix, positive_pair_logic, negative_mix, negative_pair_logic, labels

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main(args, cluster=None):

    BATCH = args.batch_size

    clean_dataset = query_gin_data(args)

    n = []
    degs = []
    for g in clean_dataset:
        num_nodes = g.num_nodes
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        n.append(g.num_nodes)
        degs.append(deg.max())
    mean_n = torch.tensor(n).float().mean().round().long().item()
    gamma = mean_n
    # p = 2 * 1 / (1 + gamma)
    p = 0.1
    # num_runs = gamma
    num_runs = 2

    def separate_data(dataset_len, seed=0):
        # Use same splitting/10-fold as GIN paper
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
            idx_list.append(idx)
        return idx_list

    class GIN(nn.Module):
        def __init__(self):
            super(GIN, self).__init__()
            num_features = clean_dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(GINConv(
                nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, clean_dataset.num_classes))
            self.fcs.append(nn.Linear(dim, clean_dataset.num_classes))

            for i in range(self.num_layers - 1):
                self.convs.append(
                    GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, clean_dataset.num_classes))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, GINConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            outs = [x]
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x)

            out = None
            for i, x in enumerate(outs):
                x = global_add_pool(x, batch)
                x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
                out, targets_a, targets_b, lam = do_noisy_mixup(x, data.y, args.alpha, args.add_noise_level, args.mult_noise_level)
                if out is None:
                    out = x
                else:
                    out += x
            return F.log_softmax(out, dim=-1), targets_a, targets_b, lam

    use_aux_loss = args.use_aux_loss

    # class DropGIN(nn.Module):
    #     def __init__(self):
    #         super(DropGIN, self).__init__()
    #
    #         num_features = clean_dataset.num_features
    #         dim = args.hidden_units
    #         self.dropout = args.dropout
    #
    #         self.num_layers = 4
    #
    #         self.convs = nn.ModuleList()
    #         self.bns = nn.ModuleList()
    #         self.fcs = nn.ModuleList()
    #
    #         self.convs.append(GINConv(
    #             nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
    #         self.bns.append(nn.BatchNorm1d(dim))
    #         self.fcs.append(nn.Linear(num_features, clean_dataset.num_classes))
    #         self.fcs.append(nn.Linear(dim, clean_dataset.num_classes))
    #
    #         for i in range(self.num_layers - 1):
    #             self.convs.append(
    #                 GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
    #             self.bns.append(nn.BatchNorm1d(dim))
    #             self.fcs.append(nn.Linear(dim, clean_dataset.num_classes))
    #
    #         if use_aux_loss:
    #             self.aux_fcs = nn.ModuleList()
    #             self.aux_fcs.append(nn.Linear(num_features, clean_dataset.num_classes))
    #             for i in range(self.num_layers):
    #                 self.aux_fcs.append(nn.Linear(dim, clean_dataset.num_classes))
    #
    #     def reset_parameters(self):
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear):
    #                 m.reset_parameters()
    #             elif isinstance(m, GINConv):
    #                 m.reset_parameters()
    #             elif isinstance(m, nn.BatchNorm1d):
    #                 m.reset_parameters()
    #
    #     def forward(self, data, flag=None):
    #         x = data.x
    #         edge_index = data.edge_index
    #         batch = data.batch
    #
    #         if flag == 0:
    #
    #             # Do runs in paralel, by repeating the graphs in the batch
    #             run_edge_index = (edge_index.repeat(1, num_runs) + torch.arange(num_runs,
    #                                                                             device=edge_index.device).repeat_interleave(
    #                 edge_index.size(1)) * (edge_index.max() + 1)).T
    #             drop = torch.bernoulli(torch.ones([run_edge_index.size(0)], device=run_edge_index.device) * p).bool()
    #             run_edge_index[drop] = torch.zeros([drop.sum().long().item(), run_edge_index.size(-1)],
    #                                                device=run_edge_index.device).long()
    #             run_edge_index = run_edge_index.T
    #             # run_edge_index = run_edge_index.view(-1, run_edge_index.size(0))
    #             x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
    #             outs = [x]
    #             x = x.view(-1, x.size(-1))
    #         else:
    #             x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
    #             drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * p).bool()
    #             x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
    #             del drop
    #             outs = [x]
    #             x = x.view(-1, x.size(-1))
    #             run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs,
    #                                                                            device=edge_index.device).repeat_interleave(
    #                 edge_index.size(1)) * (edge_index.max() + 1)
    #
    #         for i in range(self.num_layers):
    #             x = self.convs[i](x, run_edge_index)
    #             x = self.bns[i](x)
    #             x = F.relu(x)
    #             outs.append(x.view(num_runs, -1, x.size(-1)))
    #         del run_edge_index
    #         out = None
    #         # features = None
    #         for i, x in enumerate(outs):
    #             x = x.float().mean(dim=0)
    #             x = global_add_pool(x, batch)
    #             x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
    #             out, targets_a, targets_b, lam = do_noisy_mixup(x, data.y, args.alpha, args.add_noise_level,
    #                                                             args.mult_noise_level)
    #             if out is None:
    #                 out = x
    #             else:
    #                 out += x
    #
    #             # if features is None:
    #             #     features = x_feature
    #             # else:
    #             #     features += x_feature
    #
    #         if use_aux_loss:
    #             aux_out = torch.zeros(num_runs, out.size(0), out.size(1), device=out.device)
    #             run_batch = batch.repeat(num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(
    #                 batch.size(0)) * (batch.max() + 1)
    #             for i, x in enumerate(outs):
    #                 x = x.view(-1, x.size(-1))
    #                 x = global_add_pool(x, run_batch)
    #                 x_feature = x.view(num_runs, -1, x.size(-1))
    #                 x = F.dropout(self.aux_fcs[i](x_feature), p=self.dropout, training=self.training)
    #                 aux_out += x
    #             features = x_feature.mean(0)
    #
    #             return F.log_softmax(out, dim=-1), F.softmax(out, dim=-1), features, F.log_softmax(aux_out), targets_a, targets_b, lam
    #         else:
    #             return F.log_softmax(out, dim=-1), F.softmax(out, dim=-1), 0, 0,targets_a, targets_b, lam

    class DropGIN(nn.Module):
        def __init__(self):
            super(DropGIN, self).__init__()

            num_features = clean_dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(GINConv(
                nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, clean_dataset.num_classes))
            self.fcs.append(nn.Linear(dim, clean_dataset.num_classes))

            self.fc1 = torch.nn.Linear(dim, dim)
            self.fc2 = torch.nn.Linear(dim, 2)

            for i in range(self.num_layers - 1):
                self.convs.append(
                    GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, clean_dataset.num_classes))

            if use_aux_loss:
                self.aux_fcs = nn.ModuleList()
                self.aux_fcs.append(nn.Linear(num_features, clean_dataset.num_classes))
                for i in range(self.num_layers):
                    self.aux_fcs.append(nn.Linear(dim, clean_dataset.num_classes))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, GINConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            y = data.y
            edge_index = data.edge_index
            batch = data.batch

            # Do runs in paralel, by repeating the graphs in the batch
            run_edge_index = (edge_index.repeat(1, num_runs) + torch.arange(num_runs,
                                                                            device=edge_index.device).repeat_interleave(
                edge_index.size(1)) * (edge_index.max() + 1)).T
            drop = torch.bernoulli(torch.ones([run_edge_index.size(0)], device=run_edge_index.device) * p).bool()
            run_edge_index[drop] = torch.zeros([drop.sum().long().item(), run_edge_index.size(-1)],
                                               device=run_edge_index.device).long()
            run_edge_index = run_edge_index.T
            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
            outs = [x]
            x = x.view(-1, x.size(-1))

            gamma = args.gamma

            for i in range(self.num_layers):
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
            del run_edge_index
            out = None
            # features = None
            for i, x in enumerate(outs):
                x = x.float().mean(dim=0)
                x = global_add_pool(x, batch)
                x = (1+gamma) * x
                x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
                if out is None:
                    out = x
                else:
                    out += x

            if use_aux_loss:
                aux_out = torch.zeros(num_runs, out.size(0), out.size(1), device=out.device)
                run_batch = batch.repeat(num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(
                    batch.size(0)) * (batch.max() + 1)
                for i, x in enumerate(outs):
                    x = x.view(-1, x.size(-1))
                    x = global_add_pool(x, run_batch)
                    x_feature = x.view(num_runs, -1, x.size(-1))
                    x = F.dropout(self.aux_fcs[i](x_feature), p=self.dropout, training=self.training)
                    aux_out += x
                features = x_feature.mean(0)

                lam = np.random.beta(args.alpha, args.alpha) if args.alpha > 0.0 else 1.0

                '''
                '''
                perturb_idx = np.random.permutation(np.arange(y.shape[0]))
                z_plus = lam * features + (1-lam) * features[perturb_idx]
                y_plus = lam * y + (1-lam) * y[perturb_idx]
                positive_idx_list = []
                for tmp in range(y.shape[0]):
                    idx = torch.where(abs(y[tmp] - y) < 0.05)[0]
                    positive_idx_list.append(idx)

                similty = (z_plus @ z_plus.T) / torch.norm(z_plus, dim=1) / torch.norm(z_plus,dim=1).T
                weight = torch.exp(similty) / torch.exp(similty).sum(dim=1)
                hat_z = torch.zeros(z_plus.shape).to(device)
                for tmp in range(weight.shape[0]):
                    hat_z[tmp,:] = (weight[:,tmp].reshape(-1,1) * z_plus).sum(dim=0)

                features_mix_mlp = self.fc1(z_plus)
                hat_z_mlp = self.fc1(hat_z)

                pred = self.fc2(z_plus)
                cl_loss = 0
                for tmp in range(len(positive_idx_list)):
                    cl_loss += torch.dist(features_mix_mlp[tmp], features_mix_mlp[positive_idx_list[tmp]], p=2).sum() / \
                          positive_idx_list[tmp].shape[0] - torch.dist(features_mix_mlp[tmp], hat_z_mlp[tmp], p=2)

                '''
                '''
                # positive_idx = torch.zeros(y.shape).long()
                # negative_idx = torch.zeros(y.shape).long()
                #
                # one_idx = torch.where(y == 1)[0].cpu().numpy()
                # zero_idx = torch.where(y == 0)[0].cpu().numpy()
                # positive_one_index = np.random.permutation(one_idx)
                # positive_zero_idx = np.random.permutation(zero_idx)
                # positive_idx[one_idx] = torch.from_numpy(positive_one_index).long()
                # positive_idx[zero_idx] = torch.from_numpy(positive_zero_idx).long()
                # # positive pair
                # features_mix = lam * features + (1 - lam) * features[positive_idx]
                # # features_mix = features_mix.T
                # similty = (features_mix @ features_mix.T) / torch.norm(features_mix, dim=1) / torch.norm(features_mix, dim=1).T
                # weight = torch.exp(similty) / torch.exp(similty).sum(dim=1)
                # hat_z = torch.zeros(features_mix.shape).to(device)
                # for tmp in range(weight.shape[0]):
                #     # print((weight[:,tmp].reshape(-1,1) * features_mix).shape)
                #     hat_z[tmp,:] = (weight[:,tmp].reshape(-1,1) * features_mix).sum(dim=0)
                # # print(features_mix)
                # features_mix_mlp = self.fc1(features_mix)
                # hat_z_mlp = self.fc1(hat_z)
                # cl_loss = torch.dist(features_mix_mlp, features_mix_mlp[positive_idx],p=2).sum() / features_mix_mlp.shape[0]# - torch.dist(features_mix_mlp, hat_z_mlp, p=2).mean()

                # positive_pair_logic = torch.einsum('nc,nc->n', [features, features[positive_idx]]).unsqueeze(-1)
                #
                # negative_one_idx = np.random.choice(zero_idx, len(one_idx))
                # negative_zero_idx = np.random.choice(one_idx, len(zero_idx))
                # negative_idx[one_idx] = torch.from_numpy(negative_one_idx).long()
                # negative_idx[zero_idx] = torch.from_numpy(negative_zero_idx).long()
                #
                # # negative_mix = lam * features_1 + (1 - lam) * features_2[negative_idx]
                # negative_pair_logic = torch.einsum('nc,nc->n', [features, features[negative_idx]]).unsqueeze(-1)
                #
                # pair_logic = torch.cat((positive_pair_logic, negative_pair_logic),1)
                #
                # labels = torch.zeros(features.shape[0], dtype=torch.long)
                # labels[x.shape[0]:] = 1

                # features, targets_a, targets_b, lam = do_noisy_mixup(features, data.y, args.alpha, args.add_noise_level,
                #                                               args.mult_noise_level_noise_level)


                return F.log_softmax(out, dim=-1), F.softmax(out, dim=-1), features, F.log_softmax(aux_out), cl_loss, y_plus
            else:
                return F.log_softmax(out, dim=-1), F.softmax(out, dim=-1), 0, 0,0,0,0

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Device: {device}')
    if args.drop_gnn:
        model = DropGIN().to(device)
    else:
        model = GIN().to(device)
        use_aux_loss = False

    def train(epoch, loader, optimizer):
        model.train()
        loss_all = 0
        correct = 0
        all_features = torch.zeros((len(loader.dataset),args.hidden_units))
        labels = torch.zeros((len(loader.dataset)))
        probs_all = torch.zeros((len(loader.dataset), clean_dataset.num_classes))
        for i,data in enumerate(loader):
            # print(data.y)
            data = data.to(device)
            optimizer.zero_grad()
            logs, probs, features, aux_logs,cl_loss,y_plus = model(data)

            loss = F.cross_entropy(logs, abs(data.y.reshape(-1)))
            pred = logs.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            if use_aux_loss:
                aux_loss = F.cross_entropy(aux_logs.view(-1, aux_logs.size(-1)),
                                           abs(data.y.unsqueeze(0).expand(aux_logs.size(0), -1).clone().view(-1)))
                loss = 0.75 * loss + 0.25 * aux_loss
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
            all_features[i*BATCH:min((i+1)*BATCH, len(loader.dataset))] = features
            labels[i * BATCH:min((i + 1) * BATCH, len(loader.dataset))] = data.y
            probs_all[i *BATCH:min((i + 1) * BATCH, len(loader.dataset))] = probs

        return all_features, labels, probs_all, loss_all / len(loader.dataset), correct / len(loader.dataset)


    def test(loader):
        model.eval()
        loss_all = 0
        with torch.no_grad():
            correct = 0
            for data in loader:
                data = data.to(device)
                logs, probs, features, aux_logs,cl_loss,y_plus = model(data)
                pred = logs.max(1)[1]
                correct += pred.eq(data.y).sum().item()
                loss = F.cross_entropy(logs, abs(data.y.reshape(-1))) + cl_loss
                loss_all += data.num_graphs * loss.item()
        return correct / len(loader.dataset),  loss_all / len(loader.dataset)


    def label_clean(epoch, features, labels, probs, clean_label=None):
        labels = abs(labels)
        args.low_th = (1-0.1)*epoch/args.epoch
        gt_score = probs.gather(1, torch.tensor([list(labels.numpy().astype(int))]).long()).reshape(-1)
        gt_clean = gt_score > args.low_th
        soft_labels = probs.clone()
        soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), clean_dataset.num_classes).scatter_(1,labels[gt_clean].long().view(-1,1),1)

        N = features.shape[0]
        k = args.neighbor
        index = faiss.IndexFlatIP(features.shape[1])

        features_clone = features.clone()
        index.add(features_clone.detach().numpy())
        D, I = index.search(features_clone.detach().numpy(), k + 1)
        neighbors = torch.LongTensor(I)  # find k nearest neighbors excluding itself

        score = torch.zeros(N, clean_dataset.num_classes)  # holds the score from weighted-knn
        weights = torch.exp(torch.Tensor(D[:, 1:]) / args.temperature)  # weight is calculated by embeddings' similarity
        for n in range(N):
            neighbor_labels = soft_labels[neighbors[n, 1:]]
            # print(neighbor_labels.shape, weights.shape)
            score[n] = (neighbor_labels * weights[n].unsqueeze(-1)).sum(0)  # aggregate soft labels from neighbors
        soft_labels = args.lambd * score / score.sum(1).unsqueeze(
            -1) + (1-args.lambd) * probs  # combine with model's prediction as the new soft labels

        # consider the ground-truth label as clean if the soft label outputs a score higher than the threshold
        # gt_score = soft_labels[:, labels]
        gt_score = probs.gather(1, torch.tensor([list(labels.numpy().astype(int))]).long()).reshape(-1)
        gt_clean = gt_score > args.low_th
        soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), clean_dataset.num_classes).scatter_(1,labels[gt_clean].long().view(-1,1),1)

        # get the hard pseudo label and the clean subset used to calculate supervised loss
        max_score, hard_labels = torch.max(soft_labels, 1)
        clean_idx = max_score > args.high_th
        return clean_idx

    def train_pseudo(loader, optimizer, clean_idx):

        model.train()
        loss_all = 0
        correct = 0
        for i, data in enumerate(loader):
            clean_idx_batch = torch.where(clean_idx[i*BATCH:min((i+1)*BATCH, len(loader.dataset))]>0)[0]
            data = data.to(device)
            optimizer.zero_grad()
            logs, probs, features,aux_logs, cl_loss, y_plus = model(data)
            if args.alpha > 0:
                loss1 = F.cross_entropy(logs[clean_idx_batch], abs(data.y.reshape(-1))[clean_idx_batch])
                loss2 = cl_loss
                loss = loss1 + loss2
            else:
                loss = F.cross_entropy(logs[clean_idx_batch], abs(data.y.reshape(-1))[clean_idx_batch])
            pred = logs.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            if use_aux_loss:
                aux_loss = F.cross_entropy(aux_logs.view(-1, aux_logs.size(-1)),
                                           abs(data.y).unsqueeze(0).expand(aux_logs.size(0), -1).clone().view(-1))
                loss = 0.75 * loss + 0.25 * aux_loss
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()

        return loss_all / len(loader.dataset)

    acc = []
    splits = separate_data(len(clean_dataset), seed=0)
    for i, (train_idx, test_idx) in enumerate(splits):
        sample = int(args.noise_ratio * len(train_idx))
        noise_idx = random.sample(list(train_idx), sample)
        noise_dataset = query_gin_data(args, noise_idx)
        model.reset_parameters()
        lr = 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,
                                                    gamma=0.5)  # in GIN code 50 itters per epoch were used

        test_dataset = noise_dataset[test_idx.tolist()]
        train_dataset = noise_dataset[train_idx.tolist()]

        test_loader = DataLoader(test_dataset, batch_size=BATCH,shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

        print('---------------- Split {} ----------------'.format(i), flush=True)
        test_acc = 0
        acc_temp = []
        total_loss = []
        test_loss_total  = []
        train_acc_total = []
        for epoch in tqdm(range(1, args.epoch + 1)):
            if args.verbose or epoch == 350:
                start = time.time()
            lr = scheduler.optimizer.param_groups[0]['lr']
            if epoch < args.warm_up:
                features, labels, probs, train_loss, train_acc = train(epoch, train_loader, optimizer)
                train_acc_total.append(train_acc)
            else:
                clean_idx = label_clean(epoch, features, labels, probs)
                train_loss = train_pseudo(train_loader, optimizer, clean_idx)

            total_loss.append(train_loss)
            scheduler.step()
            test_acc , test_loss= test(test_loader)
            acc_temp.append(test_acc)
            test_loss_total.append(test_loss)
        acc.append(torch.tensor(acc_temp))

    acc = torch.stack(acc, dim=0)
    acc_mean = acc.mean(dim=0)
    best_epoch = acc_mean.argmax().item()

    print(args.dataset)
    print('Best results: Mean: {:7f}, Std: {:7f}'.format(acc[:, best_epoch].mean(), acc[:, best_epoch].std()), flush=True)
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_units', type=int, default=64) # 64 is used for social datasets (IMDB) and 16 or 32 for bio datasest (MUTAG, PTC, PROTEINS). Set tunable=False to not grid search over this (for social datasets)
    parser.add_argument('--use_aux_loss', action='store_true', default=True)
    parser.add_argument('--drop_gnn', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--gpu_jobs', action='store_true', default=False)
    parser.add_argument('--temperature', default=0.5)
    parser.add_argument('--low_th', default=0.1)
    parser.add_argument('--high_th', default=0.9)
    parser.add_argument('--warm_up',default=100)
    parser.add_argument('--epoch', default=200)
    parser.add_argument('--neighbor', default=200)
    parser.add_argument('--T', default=0.01)
    parser.add_argument('--hyper', default=0.01)
    parser.add_argument('--alpha', type=float, default=0.1, metavar='S', help='for mixup')
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--lambd', type=float, default=0.5)
    parser.add_argument('--manifold_mixup', type=int, default=0, metavar='S', help='manifold mixup (default: 0)')
    parser.add_argument('--add_noise_level', type=float, default=0., metavar='S', help='level of additive noise')
    parser.add_argument('--mult_noise_level', type=float, default=0, metavar='S',
                        help='level of multiplicative noise')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help="Options are ['MUTAG', 'PTC_MR','BZR', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI',   'NCI1','COLLAB', 'REDDIT-BINARY','REDDIT-MULTI-5K','ENZYMES','COX2','PROTEINS_full'，'COIL-RAG']")
    parser.add_argument('--noise_ratio', type=float, default=0.3)

    args = parser.parse_args()
    main(args)

    print('Finished', flush=True)
