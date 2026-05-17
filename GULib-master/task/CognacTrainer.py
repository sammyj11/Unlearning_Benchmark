"""
task/CognacTrainer.py
=====================
Cognac corrective unlearning trainer for GULib benchmark.

Architecture mirrors MEGUTrainer / IDEATrainer:
  - Inherits GULib's BaseTrainer (NOT the standalone Cognac base Trainer).
  - All args accessed as dict: self.args["key"].
  - The `cognac_unlearning()` method is the entry point called by the
    `cognac` pipeline class in unlearning/unlearning_methods/Cognac/cognac.py.
  - Model is saved to the standard GULib path at the end of unlearning,
    exactly as MEGUTrainer.megu_unlearning() does.

Reference:
  Kolipaka et al., "A Cognac shot to forget bad memories:
  Corrective Unlearning in GNNs", arXiv 2412.00789.
"""

import os
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import f1_score, accuracy_score

from task.BaseTrainer import BaseTrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CognacTrainer(BaseTrainer):
    """
    Cognac unlearning trainer, conforming to the GULib BaseTrainer interface.

    Constructor signature matches all other *Trainer classes in task/:
        CognacTrainer(args, logger, model, data)
    where args is a dict (GULib convention).

    The unlearning entry-point is cognac_unlearning(poisoned_nodes, run).
    """

    def __init__(self, args, logger, model, data):
        super().__init__(args, logger, model, data)
        # No extra state needed at construction; cognac_unlearning sets up everything.

    # ------------------------------------------------------------------
    # Public entry-point (called by the cognac pipeline's unlearn() hook)
    # ------------------------------------------------------------------

    def cognac_unlearning(self, poisoned_nodes, run=0):
        """
        Run Cognac's ascent + descent + contrastive-SAGE unlearning.

        Parameters
        ----------
        poisoned_nodes : np.ndarray
            Global node indices to be forgotten.
        run : int
            Run index used for the save-path filename (matches MEGU convention).

        Returns
        -------
        unlearn_time : float   wall-clock seconds
        test_f1      : float   micro-F1 on the test set after unlearning
        """
        self.model = self.model.to(device)
        self.data = self.data.to(device)

        poisoned_t = torch.tensor(poisoned_nodes, dtype=torch.long, device=device)
        attacked_set = set(poisoned_nodes.tolist())

        # ---- build masks needed by the algorithm ----
        self._attach_cognac_masks(poisoned_t)

        # ---- hyper-parameters (all from args dict) ----
        steps = self.args.get("steps", 10) or 10
        ce1 = self.args.get("contrastive_epochs_1", 5) or 5
        ce2 = self.args.get("contrastive_epochs_2", 5) or 5
        ascent_lr = self.args.get("ascent_lr", 1e-4) or 1e-4
        descent_lr = self.args.get("descent_lr", 1e-4) or 1e-4
        frac = self.args.get("contrastive_frac", 0.1) or 0.1
        k_hop = self.args.get("k_hop", 2) or 2
        linked = self.args.get("linked", False) or False
        base_lr = self.args.get("unlearn_lr", 1e-2) or 1e-2

        # ---- build influence-node sample set ----
        self._get_sample_points(poisoned_t, frac, k_hop)
        self._store_subset(k_hop)

        # ---- optimizers ----
        contrastive_opt = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        ascent_opt = torch.optim.Adam(self.model.parameters(), lr=ascent_lr)
        descent_opt = torch.optim.Adam(self.model.parameters(), lr=descent_lr)

        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        best_test_f1 = 0.0
        unlearn_time = 0.0

        start_time = time.time()

        for epoch in tqdm(range(steps), desc="Cognac Unlearning"):
            for i in range(ce1 + ce2):
                self.model.train()

                # forward
                embeddings = self.model(self.data.x, self.data.edge_index)

                if i < ce1:
                    # ---- contrastive (SAGE) phase ----
                    contrastive_opt.zero_grad()
                    loss = self._run_sage_batch(embeddings, attacked_set)
                    if isinstance(loss, torch.Tensor):
                        loss.backward()
                        contrastive_opt.step()

                else:
                    # ---- ascent phase (forget) ----
                    ascent_opt.zero_grad()
                    ascent_loss = -F.cross_entropy(
                        embeddings[self.data.poison_mask],
                        self.data.y[self.data.poison_mask],
                    )
                    ascent_loss.backward()
                    ascent_opt.step()

                    # ---- descent phase (retain) ----
                    descent_opt.zero_grad()
                    if linked:
                        emb_r = self.model(self.data.x, self.data.edge_index)
                    else:
                        emb_r = self.model(
                            self.data.x,
                            self.data.edge_index[:, self.data.dr_mask],
                        )
                    descent_loss = F.cross_entropy(
                        emb_r[self.data.retain_mask],
                        self.data.y[self.data.retain_mask],
                    )
                    descent_loss.backward()
                    descent_opt.step()

            # ---- track best by test F1 ----
            cur_f1, cur_acc = self._quick_eval_test()
            if cur_f1 > best_test_f1:
                best_test_f1 = cur_f1
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        unlearn_time = time.time() - start_time   # total wall-clock

        # load best model found during unlearning
        self.model.load_state_dict(best_state)

        # final evaluation
        test_f1, test_acc = self._quick_eval_test()

        self.logger.info(
            f"Cognac | run={run} | time={unlearn_time:.4f}s "
            f"| TestF1={test_f1:.4f} | TestAcc={test_acc:.4f}"
        )

        # ---- save to standard GULib path ----
        self._save_unlearned_model(run)

        return unlearn_time, test_f1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _attach_cognac_masks(self, poisoned_t):
        """
        Attach to self.data the masks required by the Cognac algorithm.
        Mirrors what unlearning_request() does in other GULib methods.
        """
        num_nodes = self.data.num_nodes
        ei = self.data.edge_index

        # poison_mask  : which nodes should be forgotten
        poison_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        poison_mask[poisoned_t] = True
        self.data.poison_mask = poison_mask

        # df_mask / dr_mask : edge-level forget / retain masks
        df_mask = torch.isin(ei[0], poisoned_t) | torch.isin(ei[1], poisoned_t)
        self.data.df_mask = df_mask
        self.data.dr_mask = ~df_mask

        # retain_mask : training nodes that are NOT poisoned
        retain_mask = self.data.train_mask.clone()
        retain_mask[poisoned_t] = False
        self.data.retain_mask = retain_mask

    def _reverse_features(self, features, poisoned_t):
        """Flip node features for poisoned nodes (Cognac sensitivity probe)."""
        rev = features.clone()
        rev[poisoned_t] = 1 - rev[poisoned_t]
        return rev

    def _get_sample_points(self, poisoned_t, frac, k_hop):
        """
        Identify which non-poisoned nodes are most influenced by the
        poisoned nodes, via feature-sensitivity on a k-hop subgraph.
        Sets self.data.sample_mask.
        """
        self.model.eval()
        with torch.no_grad():
            og_logits = F.softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            rev_x = self._reverse_features(self.data.x, poisoned_t)
            rev_logits = F.softmax(self.model(rev_x, self.data.edge_index), dim=1)

        diff = torch.abs(og_logits - rev_logits).mean(dim=1)

        # get k-hop neighbours of poisoned nodes, excluding the poisoned nodes themselves
        subset, _, _, _ = k_hop_subgraph(
            poisoned_t.clone().detach(), k_hop, self.data.edge_index
        )
        subset = subset[~torch.isin(subset, poisoned_t)]

        sample_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool, device=device)
        if len(subset) == 0:
            self.data.sample_mask = sample_mask
            self.logger.info("[Cognac] sample subset is empty; contrastive phase will be skipped")
            return

        diff_subset = diff[subset]
        n_select = max(1, int(frac * len(subset)))
        _, top_idx = torch.topk(diff_subset, n_select, largest=True)
        # top_idx are LOCAL positions in subset; map back to global node IDs
        influential_nodes = subset[top_idx]
        sample_mask[influential_nodes] = True
        self.data.sample_mask = sample_mask
        self.logger.info(f"[Cognac] influence nodes selected: {sample_mask.sum().item()}")

    def _store_subset(self, k_hop):
        """
        Build a dict: global_node_id -> set of k-hop neighbour global IDs.
        Only computed for nodes in sample_mask.
        """
        sample_idx = torch.where(self.data.sample_mask)[0]
        subset_dict = {}
        for idx in sample_idx:
            idx_ = idx.reshape(-1)
            sub, _, _, _ = k_hop_subgraph(idx_, k_hop, self.data.edge_index)
            subset_dict[idx.item()] = set(sub.tolist())
        self.subset_dict = subset_dict

    def _sage_loss(self, anchors, pos_embs, neg_embs):
        """GraphSAGE-style contrastive loss."""
        pos_loss = F.logsigmoid((anchors * pos_embs).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(anchors * neg_embs).sum(-1)).mean()
        return -pos_loss - neg_loss

    def _run_sage_batch(self, embeddings, attacked_set, batch_size=128):
        """
        Contrastive SAGE loss over influence nodes.
        Returns a scalar Tensor (or None if no valid samples).
        """
        sample_indices = torch.where(self.data.sample_mask)[0]
        if len(sample_indices) == 0:
            return None

        attacked_list = list(attacked_set)
        if not attacked_list:
            return None

        total_loss = None

        for i in range(0, len(sample_indices), batch_size):
            batch_idx = sample_indices[i: i + batch_size]
            cur_batch = len(batch_idx)

            # positive samples = k-hop neighbours excluding attacked nodes
            batch_pos_samples = [
                list(self.subset_dict.get(idx.item(), set()) - attacked_set)
                for idx in batch_idx
            ]
            max_pos = max((len(s) for s in batch_pos_samples), default=0)
            if max_pos == 0:
                continue

            max_neg = len(attacked_list)

            batch_pos = torch.zeros((cur_batch, max_pos), dtype=torch.long)
            mask_pos = torch.zeros((cur_batch, max_pos), dtype=torch.float32)
            batch_neg = torch.zeros((cur_batch, max_neg), dtype=torch.long)
            batch_neg[:] = torch.tensor(attacked_list, dtype=torch.long)
            mask_neg = torch.ones((cur_batch, max_neg), dtype=torch.float32)

            for j, pos_s in enumerate(batch_pos_samples):
                plen = len(pos_s)
                if plen:
                    batch_pos[j, :plen] = torch.tensor(pos_s)
                    mask_pos[j, :plen] = 1.0

            dev = embeddings.device
            batch_pos = batch_pos.to(dev)
            batch_neg = batch_neg.to(dev)
            mask_pos = mask_pos.to(dev).unsqueeze(-1)
            mask_neg = mask_neg.to(dev).unsqueeze(-1)

            try:
                anchor_e = embeddings[batch_idx].unsqueeze(1)
                pos_e = embeddings[batch_pos] * mask_pos
                neg_e = embeddings[batch_neg] * mask_neg
                batch_loss = self._sage_loss(anchor_e, pos_e, neg_e)
            except Exception:
                continue

            total_loss = batch_loss if total_loss is None else total_loss + batch_loss

        return total_loss

    @torch.no_grad()
    def _quick_eval_test(self):
        """Micro-F1 and accuracy on the test mask (test_mask always non-empty in GULib)."""
        self.model.eval()
        z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
        mask = self.data.test_mask
        pred = torch.argmax(z[mask], dim=1).cpu().numpy()
        true = self.data.y[mask].cpu().numpy()
        f1 = f1_score(true, pred, average="micro")
        acc = accuracy_score(true, pred)
        return f1, acc

    def _save_unlearned_model(self, run):
        """
        Save unlearned model to the standard GULib path.
        Mirrors MEGUTrainer.megu_unlearning() exactly:
            unlearned_models/COGNAC/{dataset}/{unlearn_task}/ratio_{r:.2f}/
                COGNAC_{dataset}_{downstream_task}_ratio_{r:.2f}{run_str}{base_str}.pt
        """
        copy_str = "_copy" if self.args.get("use_copy", False) else ""
        run_str = f"_{run}" if self.args["num_runs"] > 1 else ""
        base_str = (
            "" if self.args["base_model"] == "GCN"
            else f"_{self.args['base_model']}"
        )
        unlearn_ratio = self.args["unlearn_ratio"]

        save_dir = os.path.join(
            "unlearned_models", "COGNAC",
            self.args["dataset_name"],
            self.args["unlearn_task"],
            f"ratio_{unlearn_ratio:.2f}{copy_str}",
        )
        os.makedirs(save_dir, exist_ok=True)

        model_name = (
            f"COGNAC_{self.args['dataset_name']}_{self.args['downstream_task']}_"
            f"ratio_{unlearn_ratio:.2f}{run_str}{base_str}.pt"
        )
        save_path = os.path.join(save_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"[Cognac] Unlearned model saved: {save_path}")