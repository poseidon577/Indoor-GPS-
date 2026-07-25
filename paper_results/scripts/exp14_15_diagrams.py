import graphviz
import os
import sys

def main():
    print("Generating IEEE Style Diagrams (Parts 14 & 15)")
    diag_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagrams"))
    os.makedirs(diag_dir, exist_ok=True)
    
    # Common Graphviz IEEE Styling
    graph_attr = {
        'fontname': 'Times New Roman',
        'fontsize': '10',
        'dpi': '600',
        'bgcolor': 'transparent',
        'rankdir': 'LR'
    }
    node_attr = {
        'fontname': 'Times New Roman',
        'fontsize': '10',
        'shape': 'box',
        'style': 'filled',
        'fillcolor': 'white',
        'color': 'black'
    }
    edge_attr = {
        'fontname': 'Times New Roman',
        'fontsize': '9'
    }

    # 1. System Block Diagram
    dot = graphviz.Digraph('SystemBlockDiagram', format='svg')
    dot.attr(**graph_attr)
    dot.attr('node', **node_attr)
    dot.attr('edge', **edge_attr)
    
    dot.node('A', 'Digital Twin Dashboard')
    dot.node('B', 'Synchronization Layer')
    dot.node('C', 'Optimization Layer\n(Joint Adaptive Engine)')
    dot.node('D', 'Physics Engine')
    dot.node('E', 'Environment Twin')
    dot.node('F', 'Communication Twin')
    dot.node('G', 'Localization Twin')
    
    dot.edges(['AB', 'BC', 'CE', 'ED', 'DF', 'DG', 'FB', 'GB'])
    
    out_path = os.path.join(diag_dir, 'system_block_diagram')
    dot.render(out_path, cleanup=True)
    print("Generated system_block_diagram.svg")

    # 2. Pipeline Flowchart
    dot2 = graphviz.Digraph('Pipeline', format='svg')
    dot2.attr(**graph_attr)
    dot2.attr('node', **node_attr)
    dot2.attr('edge', **edge_attr)
    
    dot2.node('Env', 'Environment State Update')
    dot2.node('Phy', 'Physics Channel Computation')
    dot2.node('Adp', 'Adaptive Subcarrier/Bit Allocation')
    dot2.node('Pow', 'Water-Filling Power Allocation')
    dot2.node('Tx', 'OFDM Modulation (Tx)')
    dot2.node('Ch', 'Optical Channel Propagation')
    dot2.node('Rx', 'OFDM Demodulation (Rx)')
    dot2.node('Loc', 'Phase-based Localization')
    
    dot2.edge('Env', 'Phy')
    dot2.edge('Phy', 'Adp')
    dot2.edge('Adp', 'Pow')
    dot2.edge('Pow', 'Tx')
    dot2.edge('Tx', 'Ch')
    dot2.edge('Ch', 'Rx')
    dot2.edge('Ch', 'Loc')
    
    out_path = os.path.join(diag_dir, 'pipeline_flowchart')
    dot2.render(out_path, cleanup=True)
    print("Generated pipeline_flowchart.svg")

if __name__ == "__main__":
    main()
