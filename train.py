#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn

from affinity_module.config import get_config

import numpy as np

from dgllife.utils import EarlyStopping

from affinity_module.utils import set_random_seed, load_model
from affinity_module.utils import run_a_train_epoch, run_stat_epoch, run_an_eval_epoch
from affinity_module.utils import Collate
from affinity_module.dataset import VPLGDataset, FoldsOf_VPLGDataset

from affinity_module.protein_graph_loaders import DSSP_loader

from torch.backends import cudnn



cudnn.deterministic = True
cudnn.benchmark = False

args = get_config()
collate = Collate(args)
args['device'] = torch.device("cuda: 0") if torch.cuda.is_available() else torch.device("cpu")
set_random_seed(args['random_seed'])


argv_valFold = args['argv_valFold']
argv_testFold = args['argv_testFold']
cache_dir_prefix = args['cache_dir_prefix']

pdb2graph_translator = DSSP_loader(dssp_files_path = args['dssp_files_path'],
                                   includeAminoacidPhyschemFeatures = False,
                                   cache_dir_prefix = cache_dir_prefix)

best_model_filename = pdb2graph_translator.get_best_model_filename()

_colNames = dict(master_data_table = args['master_data_table'],
                 pdb_id_col_name="PDBs", smiles_col_name="SMILES", target_col_name="logKi",
                 foldId_col_name='Fold')

dataset = VPLGDataset(
    smiles_to_graph=args['smiles_to_graph'],
    smiles_node_featurizer=args['smiles_node_featurizer'],
    smiles_edge_featurizer=args['smiles_edge_featurizer'],
    **_colNames,
    pdb2graph_translator = pdb2graph_translator,
    load=False)



args['device'] = torch.device("cuda: 0") if torch.cuda.is_available() else torch.device("cpu")

raw_dataset = VPLGDataset(
    smiles_to_graph=args['smiles_to_graph'],
    smiles_node_featurizer=args['smiles_node_featurizer'],
    smiles_edge_featurizer=args['smiles_edge_featurizer'],
    **_colNames,
    pdb2graph_translator = pdb2graph_translator,
    load=True)


args['fasta_node_feat_size'] = raw_dataset.fasta_graphs[0].ndata['h'].shape[1]
args['fasta_edge_feat_size'] = raw_dataset.fasta_graphs[0].edata['e'].shape[1]
print(args['fasta_node_feat_size'], args['fasta_edge_feat_size'])

print('will save best model to:', best_model_filename)
shfl = True

p = dict(batch_size=args['batch_size'], shuffle=shfl, collate_fn=collate.collate_molgraphs)

mx_nodes = 5000 # filter out large graphs
folds5 = [0,1,2,3,4]
folds5.remove(argv_valFold)
if argv_testFold in folds5:
    folds5.remove(argv_testFold)
print('Folds to use: train=%s, val=%s, test=%s' % (str(folds5), str(argv_valFold), str(argv_testFold)) )
#
train_loader = FoldsOf_VPLGDataset(raw_dataset, folds5, max_nodes = mx_nodes).asDataLoader(**p)
val_loader =   FoldsOf_VPLGDataset(raw_dataset, [argv_valFold], max_nodes = mx_nodes ).asDataLoader(**p)
test_loader =  FoldsOf_VPLGDataset(raw_dataset, [argv_testFold], max_nodes = mx_nodes ).asDataLoader(**p)
print('number of batches: train %d, val %d, test %d' % (len(train_loader), len(val_loader), len(test_loader)))

torch.cuda.empty_cache()

model = load_model(args)
loss_fn = nn.MSELoss(reduction='none')

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                             weight_decay=args['weight_decay'])

stopper = EarlyStopping(mode=args['mode'],
                        patience=args['patience'],
                        filename=best_model_filename)

if args['load_checkpoint']:
    print('Loading checkpoint...')
    stopper.load_checkpoint(model)
model.to(args['device'])

for epoch in range(args['num_epochs']):
    # Train
    run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

    # Validation and early stop
    val_score_ext = run_stat_epoch(args, model, val_loader)
    val_score = val_score_ext[ args['metric_name'] ]

    test_score = run_an_eval_epoch(args, model, test_loader)
    early_stop = stopper.step(val_score, model)
    print('epoch {:d}/{:d}, validation {} {:.4f}, test {} {:.4f}, best validation {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric_name'], val_score,
        args['metric_name'], test_score,
        args['metric_name'], stopper.best_score),
          ', now R2 = %.4f' % val_score_ext['R2'])

    if early_stop:
        break

print('-'*80)
stopper.load_checkpoint(model)

print()

all_metrics = {}
for dsName, data_loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
    metrics, _, y_true, y_pred = run_stat_epoch(args, model, data_loader, return_pred=True)
    all_metrics[dsName] = (metrics, y_true, y_pred)


print('-'*50)


all_metric_names = set()
for dsName in ['train', 'val', 'test']:
    metrics, _, _ = all_metrics[dsName]
    for k in metrics.keys():
        all_metric_names.add(k)
#

print('%25s' % '', end='')
for dsName in ['train', 'val', 'test']:
    print('%12s' % dsName, end='')
print()
for mName in all_metric_names:
    print('%25s' % mName, end='')
    for dsName in ['train', 'val', 'test']:
        metrics, _, _ = all_metrics[dsName]
        print('%12.5f' % metrics[mName], end='')
    print()
#
print('-'*50)




_baseFname = pdb2graph_translator.get_best_model_filename().replace('.pth', '')

for dsName in ['train', 'val', 'test']:
    _, y_true, y_pred = all_metrics[dsName]
    y_true = np.array(y_true)[:, 0]
    y_pred = np.array(y_pred)

    np.savetxt(_baseFname+'_yy_%s.txt' % dsName, np.vstack((y_true, y_pred)).T, 
               header='y_true, y_pred (%s)' % dsName )

