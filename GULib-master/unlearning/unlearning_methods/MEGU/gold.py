import numpy as np
from unlearning.unlearning_methods.MEGU.megu import megu
from config import unlearning_path, unlearning_edge_path


class gold(megu):
    """
    Gold-standard unlearning method: retrain from scratch on the remaining data.

    This class reuses the full MEGU pipeline (model init, original training)
    but owns its own ``unlearning_request`` so that ``gold_data`` is always
    built here, independent of whichever version of ``megu.unlearning_request``
    is present.  The ``unlearn`` step calls only ``gold_standard_unlearning``;
    it never touches ``megu_unlearning``.

    Usage::

        python main.py --unlearning_methods GOLD ...
    """

    def unlearning_request(self):
        """
        Prepare the graph data for gold-standard unlearning.

        Mirrors the data-preparation logic of ``megu.unlearning_request``
        (loading the forget set, updating edge/feature tensors) and always
        creates ``self.gold_data`` via ``_create_gold_standard_data`` at the
        end, regardless of the parent class version.
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
            self.unlearing_nodes = unique_nodes
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)

        elif self.args["unlearn_task"] == "edge":
            path_un = (
                unlearning_edge_path
                + "_" + str(self.run)
                + "_edges_" + str(self.args["num_unlearned_edges"])
                + ".txt"
            )
            remove_edges = np.loadtxt(path_un, dtype=int)
            unique_nodes = np.unique(remove_edges)
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(
                unique_nodes, remove_edges
            )

        elif self.args["unlearn_task"] == "feature":
            path_un = (
                unlearning_path
                + "_" + str(self.run)
                + "_nodes_" + str(self.args["num_unlearned_nodes"])
                + ".txt"
            )
            unique_nodes = np.loadtxt(path_un, dtype=int)
            self.unlearing_nodes = unique_nodes
            self.data.x_unlearn[unique_nodes] = 0.0

        else:
            raise ValueError(
                f"Unknown unlearn_task '{self.args['unlearn_task']}'. "
                "Expected one of: 'node', 'edge', 'feature'."
            )

        self.temp_node = unique_nodes
        self.target_model.data = self.data

        # Always build gold_data here — this is the sole purpose of this class.
        self.gold_data = self._create_gold_standard_data()

    def unlearn(self):
        """
        Run gold-standard unlearning only (retrain from scratch).

        Records timing and F1 in ``avg_unlearning_time_gold`` /
        ``average_f1_gold``.  The MIA attack inside
        ``gold_standard_unlearning`` is gated by ``self.args["attack"]``.
        """
        self.avg_unlearning_time_gold[self.run], self.average_f1_gold[self.run] = (
            self.gold_standard_unlearning()
        )