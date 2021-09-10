import numpy as np
import torch
import random
import dgl
from dgllife.utils import Meter
from affinity_module.config import get_config

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, median_absolute_error, r2_score
from scipy.stats import median_absolute_deviation


class Collate:
    def __init__(self, conf):
        self.config = conf

    def collate_molgraphs(self, data):
        if self.config['inference']:
            fasta_graphs, smiles, graphs = map(list, zip(*data))
        else:
            fasta_graphs, smiles, graphs, labels = map(list, zip(*data))

        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        fbg = dgl.batch(fasta_graphs)
        fbg.set_n_initializer(dgl.init.zero_initializer)
        fbg.set_e_initializer(dgl.init.zero_initializer)

        if not self.config['inference']:
            labels = torch.stack(labels, dim=0)

        if self.config['inference']:
            return fbg, smiles, bg
        else:
            return fbg, smiles, bg, labels


def regress(args, model, bg, fbg, get_weights=False):
    atom_feats, bond_feats = bg.ndata.pop('h'), bg.edata.pop('e')

    atom_feats, bond_feats = atom_feats.to(args['device']), bond_feats.to(args['device'])

    plg_node_feats, plg_edge_feats = fbg.ndata.pop('h'), fbg.edata.pop('e')
    plg_node_feats, plg_edge_feats = plg_node_feats.to(args['device']), plg_edge_feats.to(args['device'])

    return model(bg, fbg, atom_feats, bond_feats, plg_node_feats, plg_edge_feats, get_weights)


def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        fbg, smiles, bg, labels = batch_data
        labels = labels.to(args['device'])
        if len(labels)==1:
            print('  Will ignore batch_id', batch_id, 'because its size == 1 batch is incompatible with BatchNorm1d in  MLPPredictor')
            print(smiles, labels)
            continue

        prediction = regress(args, model, bg, fbg)#, skip_computation = _skip)
        loss = loss_criterion(prediction, labels).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    total_score = np.percentile(train_meter.compute_metric(args['metric_name']), 90)
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric_name'], total_score))


def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            fbg, smiles, bg, labels = batch_data
            if len(labels)==1:
                print('  [run_an_eval_epoch] will ignore batch', batch_id, 'with size 1')
                continue # otherwise one of the graphs gets empty after calling regress() o_O

            labels = labels.to(args['device'])
            prediction = regress(args, model, bg, fbg)
            eval_meter.update(prediction, labels)
        total_score = np.percentile(eval_meter.compute_metric(args['metric_name']), 90)
    return total_score


def run_stat_epoch(args, model, data_loader, return_pred=False):
    model.eval()
    all_smiles = []
    all_labels = []
    all_predictions = []
    all_devs = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            fbg, smiles, bg, labels = batch_data
            if len(labels)==1:
                print('  [run_stat_epoch] will ignore batch', batch_id, 'with size 1')
                continue # otherwise one of the graphs gets empty after calling regress() o_O

            labels = labels.to(args['device'])
            prediction = regress(args, model, bg, fbg).view(-1)
            dev = abs(labels - prediction)

            all_smiles.extend(smiles)
            all_labels.extend(labels.tolist())
            all_predictions.extend(prediction.tolist())
            all_devs.extend(dev.tolist())

        mse = mean_squared_error(all_labels, all_predictions)
        mae = mean_absolute_error(all_labels, all_predictions)
        max_err = max_error(all_labels, all_predictions)
        median_absolute_err = median_absolute_error(all_labels, all_predictions)
        r2 = r2_score(all_labels, all_predictions)
    metrics_dict = {'mse': mse,
                'mae': mae,
                'max_error': max_err,
                'median_absolute_error': median_absolute_err,
                'R2': r2
               }
    if return_pred:
        return metrics_dict, all_smiles, all_labels, all_predictions
    else:
        return metrics_dict


def run_an_inference_epoch(args, model, data_loader):
    model.eval()
    predictions_by_batch = []
    fastas = []
    smiless = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            fasta, fbg, smiles, bg = batch_data
            prediction = regress(args, model, bg, fbg)
            predictions_by_batch.append(prediction)
            fastas.append(fasta)
            smiless.append(smiles)
    return fastas, smiless, predictions_by_batch


def set_random_seed(seed=0):
    """Set random seed.
       seed : int --  Random seed to use. Default to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # enables/disables the cudnn auto-tuner to find the best algorithm to use.
                                                # There are several algorithms without reproducibility guarantees. [SOverfl.]
        torch.backends.cudnn.deterministic = True


def load_model(args):
    if args['model'] == 'mhGANN':
        from affinity_module.model import mhGANN
        model = mhGANN(
                       smiles_node_feat_size=args['smiles_node_featurizer'].feat_size(),
                       smiles_edge_feat_size=args['smiles_edge_featurizer'].feat_size(),
                       fasta_node_feat_size=args['fasta_node_feat_size'],
                       fasta_edge_feat_size=args['fasta_edge_feat_size'],
                       smiles_num_layers=args['smiles_num_layers'],
                       smiles_num_timesteps=args['smiles_num_timesteps'],
                       smiles_graph_feat_size=args['smiles_graph_feat_size'],
                       fasta_num_layers=args['fasta_num_layers'],
                       fasta_num_timesteps=args['fasta_num_timesteps'],
                       fasta_graph_feat_size=args['fasta_graph_feat_size'],
                       n_tasks=args['n_tasks'],
                       dropout=args['dropout'],
                       mlp_hidden_layer=args['mlp_hidden_layer'])

    return model
