"""
unlearning/unlearning_methods/Cognac/cognac.py
===============================================
Cognac unlearning pipeline for GULib.

Structure mirrors megu.py exactly:
  - Inherits Learning_based_pipeline.
  - determine_target_model() sets args["unlearn_trainer"] = "CognacTrainer"
    and calls get_trainer() — same as MEGU sets "MEGUTrainer".
  - train_original_model() uses _train_model() helper — same as MEGU.
  - unlearning_request() reads the standard text-file unlearning indices
    (same path convention as MEGU / GIF / IDEA).
  - unlearn() calls self.target_model.cognac_unlearning(temp_node, run)
    — same as MEGU calls self.target_model.megu_unlearning(temp_node, ...).

Reference:
  Kolipaka et al., "A Cognac shot to forget bad memories:
  Corrective Unlearning in GNNs", arXiv 2412.00789.
"""

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from pipeline.Learning_based_pipeline import Learning_based_pipeline
from task import get_trainer
from config import BLUE_COLOR, RESET_COLOR
from config import unlearning_path, unlearning_edge_path



class cognac(Learning_based_pipeline):
    """
    Cognac: Corrective Unlearning in GNNs.

    Registered in unlearning_manager.py as:
        method_map["COGNAC"] = cognac

    CLI usage:
        python GULib-master/main.py --unlearning_methods COGNAC ...
    """

    def __init__(self, args, logger, model_zoo):
        super().__init__(args, logger, model_zoo)
        self.args = args
        self.logger = logger
        self.model_zoo = model_zoo
        self.data = self.model_zoo.data
        self._data = copy.deepcopy(self.data)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_feats = self.data.num_features

        num_runs = self.args["num_runs"]
        self.run = 0
        self.average_f1 = np.zeros(num_runs)
        self.average_auc = np.zeros(num_runs)
        self.avg_unlearning_time = np.zeros(num_runs)
        self.avg_training_time = np.zeros(num_runs)

        # Store index arrays for MIA (same pattern as MEGU)
        self.train_indices = self.data.train_indices
        self.test_indices = self.data.test_indices

    # ----------------------------------------------------------------
    # Hook 1 — mirrors megu.determine_target_model()
    # ----------------------------------------------------------------
    def determine_target_model(self):
        """Construct a CognacTrainer wrapping the base GNN model."""
        self.logger.info("target model: %s" % self.args["base_model"])
        self.args["unlearn_trainer"] = "CognacTrainer"
        self.target_model = get_trainer(
            self.args, self.logger, self.model_zoo.model, self._data
        )

    # ----------------------------------------------------------------
    # Hook 2 — mirrors megu.train_original_model()
    # ----------------------------------------------------------------
    def train_original_model(self):
        """Train (or load from disk) the base GNN on the clean graph."""
        self.logger.info("training target models, run %s" % self.run)
        run_training_time, _ = self._train_model(self.run)
        self.avg_training_time[self.run] = run_training_time

    # ----------------------------------------------------------------
    # Hook 3 — mirrors megu.unlearning_request()
    # ----------------------------------------------------------------
    def unlearning_request(self):
        """
        Read the standard GULib unlearning text-file (same path as MEGU)
        and store the unlearning node indices in self.temp_node.
        Also updates self.target_model.data so the trainer sees the
        unlearning request (same pattern MEGU uses with:
            self.target_model.data = self.data
        ).
        """
        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()

        if self.args["unlearn_task"] == "node":
            path_un = (
                unlearning_path
                + "_" + str(self.run)
                + "_nodes_" + str(self.args["num_unlearned_nodes"])
                + ".txt"
            )
            unique_nodes = np.loadtxt(path_un, dtype=int)
            if unique_nodes.ndim == 0:
                unique_nodes = np.array([unique_nodes.item()])
            self.unlearing_nodes = unique_nodes       # MEGU naming convention
            self.temp_node = unique_nodes

        elif self.args["unlearn_task"] == "edge":
            path_un = (
                unlearning_edge_path
                + "_" + str(self.run)
                + "_edges_" + str(self.args["num_unlearned_edges"])
                + ".txt"
            )
            remove_edges = np.loadtxt(path_un, dtype=int)
            if remove_edges.ndim == 0:
                remove_edges = np.array([remove_edges.item()])
            unique_nodes = np.unique(remove_edges)
            self.unlearing_nodes = unique_nodes
            self.temp_node = unique_nodes

        elif self.args["unlearn_task"] == "feature":
            path_un = (
                unlearning_path
                + "_" + str(self.run)
                + "_nodes_" + str(self.args["num_unlearned_nodes"])
                + ".txt"
            )
            unique_nodes = np.loadtxt(path_un, dtype=int)
            if unique_nodes.ndim == 0:
                unique_nodes = np.array([unique_nodes.item()])
            self.unlearing_nodes = unique_nodes
            self.temp_node = unique_nodes
            self.data.x_unlearn[unique_nodes] = 0.0  # zero-out features

        else:
            raise ValueError(
                f"[Cognac] unsupported unlearn_task: {self.args['unlearn_task']}"
            )

        # Hand modified data to the trainer (same pattern as MEGU)
        self.target_model.data = self.data

    # ----------------------------------------------------------------
    # Hook 4 — mirrors megu.unlearn()
    # ----------------------------------------------------------------
    def unlearn(self):
        """
        Call CognacTrainer.cognac_unlearning().

        FIX (Bug 2): Capture the original model's softlabels BEFORE
        cognac_unlearning() modifies self.target_model.model in-place.
        mia_attack() (called by run_exp() after this method) then uses
        self.original_softlabels vs the post-unlearning outputs.

        FIX (Bug 1): Do NOT call self.mia_attack() here.
        Learning_based_pipeline.run_exp() already calls it after unlearn().
        Calling it here too would run MIA twice and overwrite average_auc[run].
        """
        # Capture original softlabels BEFORE unlearning (Bug 2 fix)
        self.data = self.data.to(self.device)   # ensure data is on same device as model
        self.target_model.model.eval()
        with torch.no_grad():
            self.original_softlabels = F.softmax(
                self.target_model.model(self.data.x, self.data.edge_index), dim=1
            ).clone().detach().float()

        unlearn_time, test_f1 = self.target_model.cognac_unlearning(
            self.temp_node, run=self.run
        )
        self.avg_unlearning_time[self.run] = unlearn_time
        self.average_f1[self.run] = test_f1

        self.logger.info(
            "%sCognac Performance | run=%d | TestF1=%.4f | UnlearnTime=%.4f s%s"
            % (BLUE_COLOR, self.run, test_f1, unlearn_time, RESET_COLOR)
        )
        # NOTE: mia_attack() is NOT called here.
        # Learning_based_pipeline.run_exp() calls it after unlearn() when attack=True.

    # ----------------------------------------------------------------
    # Helpers — mirrors MEGU's _train_model() / mia_attack()
    # ----------------------------------------------------------------
    def _train_model(self, run):
        """Thin wrapper matching MEGU's _train_model()."""
        start_time = time.time()
        res = self.target_model.train()
        train_time = time.time() - start_time
        return train_time, res

    def mia_attack(self):
        """
        Membership-inference attack mirroring MEGU's mia_attack().
        Uses self.original_softlabels (saved in unlearn() BEFORE
        cognac_unlearning ran) vs the post-unlearning model outputs.
        """
        try:
            mia_num = self.unlearing_nodes.shape[0]
            if mia_num > len(self.data.test_indices):
                mia_num = len(self.data.test_indices)

            # original_softlabels captured in unlearn() before weights changed
            original_softlabels = self.original_softlabels

            # post-unlearning softlabels from the now-modified model
            self.target_model.model.eval()
            with torch.no_grad():
                unlearn_softlabels = F.softmax(
                    self.target_model.model(self.data.x, self.data.edge_index), dim=1
                ).clone().detach().float()

            orig_member = original_softlabels[self.unlearing_nodes[:mia_num]]
            orig_non    = original_softlabels[self.data.test_indices[:mia_num]]
            unl_member  = unlearn_softlabels[self.unlearing_nodes[:mia_num]]
            unl_non     = unlearn_softlabels[self.data.test_indices[:mia_num]]

            mia_test_y = torch.cat((torch.ones(mia_num), torch.zeros(mia_num)))
            posterior1 = torch.cat((orig_member, orig_non), 0).cpu().detach()
            posterior2 = torch.cat((unl_member, unl_non), 0).cpu().detach()
            posterior = np.array(
                [np.linalg.norm(posterior1[i] - posterior2[i]) for i in range(len(posterior1))]
            )
            auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
            self.average_auc[self.run] = auc
            return auc
        except Exception as e:
            self.logger.warning(f"[Cognac] MIA attack skipped: {e}")
            return 0.0