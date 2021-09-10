import torch

import torch.nn as nn

from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from dgllife.model.model_zoo.mlp_predictor import MLPPredictor


__all__ = ['mhGANN']


class mhGANN(nn.Module):
    """AttentiveFP for regression and classification on graphs.
    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    """
    def __init__(self,
                 smiles_node_feat_size,
                 smiles_edge_feat_size,
                 fasta_node_feat_size,
                 fasta_edge_feat_size,
                 smiles_num_layers=2,
                 smiles_num_timesteps=2,
                 smiles_graph_feat_size=200,
                 fasta_num_layers=2,
                 fasta_num_timesteps=2,
                 fasta_graph_feat_size=200,
                 n_tasks=1,
                 dropout=0.,
                 mlp_hidden_layer=256):
        super(mhGANN, self).__init__()

        self.smiles_gnn = AttentiveFPGNN(node_feat_size=smiles_node_feat_size,
                                          edge_feat_size=smiles_edge_feat_size,
                                          num_layers=smiles_num_layers,
                                          graph_feat_size=smiles_graph_feat_size,
                                          dropout=dropout)

        self.fasta_gnn = AttentiveFPGNN(node_feat_size=fasta_node_feat_size,
                                          edge_feat_size=fasta_edge_feat_size,
                                          num_layers=fasta_num_layers,
                                          graph_feat_size=fasta_graph_feat_size,
                                          dropout=dropout)

        self.smiles_readout = AttentiveFPReadout(feat_size=smiles_graph_feat_size,
                                          num_timesteps=smiles_num_timesteps,
                                          dropout=dropout)

        self.fasta_readout = AttentiveFPReadout(feat_size=fasta_graph_feat_size,
                                          num_timesteps=fasta_num_timesteps,
                                          dropout=dropout)

        self.predict = MLPPredictor(
                             smiles_graph_feat_size+fasta_graph_feat_size,  # input layer size
                             smiles_graph_feat_size+fasta_graph_feat_size,  # 400,  # hidden layer size
                             n_tasks, dropout)
    #

    def forward(self, smiles_g, fasta_g,
                smiles_node_feats, smiles_edge_feats,
                plg_node_feats, plg_edge_feats, get_node_weight=False):
        """ Graph-level regression
        """
        smiles_node_feats = self.smiles_gnn(smiles_g, smiles_node_feats, smiles_edge_feats)
        plg_node_feats = self.fasta_gnn(fasta_g, plg_node_feats, plg_edge_feats)
        if get_node_weight:
            smiles_g_feats, smiles_node_weights = self.smiles_readout(smiles_g, smiles_node_feats, get_node_weight)
            plg_g_feats, fasta_node_weights = self.fasta_readout(fasta_g, plg_node_feats, get_node_weight)
            g_feats = torch.cat((smiles_g_feats, plg_g_feats), dim=1)
            return self.predict(g_feats), smiles_node_weights, fasta_node_weights
        else:
            smiles_g_feats = self.smiles_readout(smiles_g, smiles_node_feats)
            plg_g_feats = self.fasta_readout(fasta_g, plg_node_feats)
            g_feats = torch.cat((smiles_g_feats, plg_g_feats), dim=1)
            return self.predict(g_feats)
