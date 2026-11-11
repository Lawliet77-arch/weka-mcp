package weka.mcp.gui;

import tools.jackson.databind.node.ObjectNode;

final class ExplorerRunState {
    final String id;
    final ObjectNode dataset;
    final int historyAtStart;
    boolean cancelled;

    ExplorerRunState(String id, ObjectNode dataset, int historyAtStart) {
        this.id = id;
        this.dataset = dataset;
        this.historyAtStart = historyAtStart;
    }
}
