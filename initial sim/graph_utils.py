import networkx as nx
from collections import deque
from matplotlib import pyplot as plt


def follow_alternating_path(graph, start_node):
    """
    Follow an alternating path from the given start node until reaching a nash equilibrium or a visited non-nash node

    * If the terminal node is not nash, then the graph contains an "alternating cycle" in which agents will take actions without convergence indefinitely
    """
    assert all([data.get("agent", None) for _, _, data in graph.edges(data=True)]), "all edges must have agent attribute"

    current_node = start_node
    last_agent = None
    visited = set()

    while (current_node, last_agent) not in visited:
        visited.add((current_node, last_agent))

        # Find next edge based on alternating agent logic
        outgoing_edges = list(graph.out_edges(current_node, data="agent"))
        next_edge = None
        for edge in outgoing_edges:
            _, next_node, agent = edge
            if agent != last_agent:  # Ensure alternation
                next_edge = (next_node, agent)
                break

        if not next_edge:  # No valid edge, stop
            break

        next_node, last_agent = next_edge
        current_node = next_node

    return current_node  # Return terminal node


def find_nash_equilibrium_nodes(policy_graph):
    """
    Finds all nodes in a directed graph where all outgoing edges point to the node itself.

    Args:
        G (nx.DiGraph): A directed graph.

    Returns:
        list: A list of self-equilibrium nodes.
    """
    nash_equilibrium_nodes = []
    for node in policy_graph.nodes:
        out_neighbors = list(policy_graph.successors(node))  # Get all nodes this node points to
        if len(out_neighbors) > 0 and set(out_neighbors) == {node}:  # All point to itself
            nash_equilibrium_nodes.append(node)
    return nash_equilibrium_nodes


def compute_nash_convergence(graph: nx.MultiDiGraph, nash_nodes, agents):
    """
    Computes convergence to Nash equilibria with alternating agent turns,
    starting with any agent in the first step.

    Args:
        graph: NetworkX DiGraph with edges labeled by agent performing the best response.
        nash_nodes: Set of nodes that are Nash equilibria.
        agents: List of agent identifiers (e.g., [0, 1, ..., k-1]).

    Returns:
        A dictionary mapping each node to the Nash node it converges to.
    """
    convergence = {node: [None for _ in agents] for node in graph.nodes()}

    # Initialize convergence for Nash nodes
    for nash_node in nash_nodes:
        convergence[nash_node] = [nash_node for _ in agents]

    # Process BFS for each starting agent
    for i, starting_agent in enumerate(agents):
        queue = deque([(nash_node, starting_agent) for nash_node in nash_nodes])

        while queue:
            current_node, current_agent = queue.popleft()
            curr_agent_idx = agents.index(current_agent)
            for predecessor in graph.predecessors(current_node):
                # TODO - MultiDiGraph has key attr that is currently not used, consider using it for agent description
                if graph.edges[predecessor, current_node, 0].get('agent') == current_agent:
                    if convergence[predecessor][curr_agent_idx] is None:
                        # set convergence of predecessor action as the convergence of next action from the current node
                        # TODO - verify with more than 2 agents ...
                        convergence[predecessor][curr_agent_idx] = convergence[current_node][(curr_agent_idx + 1) % len(agents)]
                        next_agent = agents[(curr_agent_idx - 1) % len(agents)]
                        queue.append((predecessor, next_agent))

    return convergence


def plot_policy_best_response_graph(policy_graph):
    assert all(
        [data.get("color", None) for _, _, data in policy_graph.edges(data=True)]), "all edges must have color attribute"
    assert all(
        [data.get("style", None) for _, _, data in policy_graph.edges(data=True)]), "all edges must have style attribute"
    assert all(
        [data["color"] for _, data in policy_graph.nodes(data=True)]), "all nodes must have color attribute"

    # Get edge colors from attributes
    edge_colors = [data["color"] for _, _, data in policy_graph.edges(data=True)]
    edge_styles = [data["style"] for _, _, data in policy_graph.edges(data=True)]

    # node attributes
    node_colors = [data["color"] for _, data in policy_graph.nodes(data=True)]
    in_degrees = dict(policy_graph.in_degree())
    node_sizes = [(in_degrees[node] + 1) * 50 for node in policy_graph.nodes]

    # Draw the graph
    pos = nx.spring_layout(policy_graph, scale=3)  # Position nodes with a spring layout

    # Draw graph
    nx.draw(policy_graph, pos, with_labels=True,
            node_color=node_colors, node_size=node_sizes,
            edge_color=edge_colors,
            style=edge_styles, font_size=8, connectionstyle="arc3,rad=0.2")

    # Show the graph
    plt.show()