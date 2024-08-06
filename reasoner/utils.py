import sys

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from matplotlib.patches import ConnectionPatch
from itertools import groupby

from reasoner.config import NODE_TYPE_TO_INT, EDGE_TYPE_TO_INT


def is_debugging():
    return sys.gettrace() is not None


def parse_grf_file(filename, force_undirected=False, edges_have_labels=True):
    if force_undirected:
        graph = nx.Graph()  # 使用无向图表示
    else:
        graph = nx.DiGraph()  # 使用有向图表示

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if len(lines) < 1:
        raise ValueError("文件格式错误：缺少节点数量信息。")

    try:
        num_nodes = int(lines[0])
    except ValueError:
        raise ValueError("文件格式错误：节点数量不是有效的整数。")

    if len(lines) < num_nodes + 1:
        raise ValueError("文件格式错误：节点信息不完整。")

    node_lines = lines[1:num_nodes + 1]  # 节点信息部分
    edge_lines = lines[num_nodes + 1:]  # 边信息部分

    for line in node_lines:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("文件格式错误：节点信息格式不正确。")
        node_id, node_attr = map(int, parts)
        graph.add_node(node_id, attr=node_attr)

    line_index = 0
    while line_index < len(edge_lines):
        if edge_lines[line_index] == '':
            line_index += 1
            continue

        try:
            num_edges = int(edge_lines[line_index])
        except ValueError:
            raise ValueError("文件格式错误：边数量不是有效的整数。")
        line_index += 1

        for _ in range(num_edges):
            if line_index >= len(edge_lines):
                raise ValueError("文件格式错误：边信息不完整。")
            parts = edge_lines[line_index].split()
            if edges_have_labels:
                if len(parts) != 3:
                    raise ValueError("文件格式错误：边信息格式不正确，应为'源节点 目标节点 边属性'。")
                src, dst, edge_attr = map(int, parts)
                graph.add_edge(src, dst, attr=edge_attr)
            else:
                if len(parts) != 2:
                    raise ValueError("文件格式错误：边信息格式不正确，应为'源节点 目标节点'（没有边属性）。")
                src, dst = map(int, parts)
                graph.add_edge(src, dst)
            line_index += 1

    return graph


def get_global_attribute_range(graph1, graph2, attribute='attr'):
    values_graph1 = ([data.get(attribute, 0) for _, data in graph1.nodes(data=True)] +
                     [data.get(attribute, 0) for _, _, data in graph1.edges(data=True)])
    values_graph2 = ([data.get(attribute, 0) for _, data in graph2.nodes(data=True)] +
                     [data.get(attribute, 0) for _, _, data in graph2.edges(data=True)])
    global_min = min(min(values_graph1), min(values_graph2))
    global_max = max(max(values_graph1), max(values_graph2))
    return global_min, global_max


def get_node_color_map(graph, global_min, global_max, attribute='attr', color_scheme='coolwarm'):
    values = [data.get(attribute, 0) for _, data in graph.nodes(data=True)]  # Use default value if attribute not found
    cmap = plt.get_cmap(color_scheme)
    norm = plt.Normalize(global_min, global_max)
    color_map = [cmap(norm(value)) for value in values]
    return color_map


def get_edge_color_map(graph, global_min, global_max, attribute='attr', color_scheme='coolwarm'):
    values = [data.get(attribute, 0) for _, _, data in
              graph.edges(data=True)]  # Use default value if attribute not found
    cmap = plt.get_cmap(color_scheme)
    norm = plt.Normalize(global_min, global_max)
    color_map = dict(zip(graph.edges(), [cmap(norm(value)) for value in values]))
    return color_map


def visualize_graphs(pattern_graph, global_graph, mapping):
    pos_pattern = nx.spring_layout(pattern_graph, k=1)
    pos_global = nx.spring_layout(global_graph, k=1)

    # 为节点设置颜色
    global_min, global_max = get_global_attribute_range(pattern_graph, global_graph, attribute='attr')

    pattern_node_colors = get_node_color_map(pattern_graph, global_min, global_max, attribute='attr')
    global_node_colors = get_node_color_map(global_graph, global_min, global_max, attribute='attr')

    plt.figure(figsize=(16, 8))

    # Subgraph
    ax1 = plt.subplot(1, 2, 1)
    nx.draw_networkx_nodes(pattern_graph, pos=pos_pattern, node_color=pattern_node_colors, ax=ax1, node_size=750)
    nx.draw_networkx_labels(pattern_graph, pos=pos_pattern, ax=ax1, font_size=24)

    # 检查 pattern_graph 中的边是否有 'attr' 属性
    if all('attr' in pattern_graph[u][v] for u, v in pattern_graph.edges()):
        pattern_edge_colors = get_edge_color_map(pattern_graph, global_min, global_max, attribute='attr')
        nx.draw_networkx_edges(pattern_graph, pos=pos_pattern, ax=ax1,
                               edge_color=[pattern_edge_colors[edge] for edge in pattern_graph.edges()])
    else:
        nx.draw_networkx_edges(pattern_graph, pos=pos_pattern, ax=ax1, edge_color='gray')

    ax1.set_title('Subgraph', fontsize=24)

    # Global Graph
    ax2 = plt.subplot(1, 2, 2)
    nx.draw_networkx_nodes(global_graph, pos=pos_global, node_color=global_node_colors, ax=ax2, node_size=750)
    nx.draw_networkx_labels(global_graph, pos=pos_global, ax=ax2, font_size=24)

    # 检查 global_graph 中的边是否有 'attr' 属性
    if all('attr' in global_graph[u][v] for u, v in global_graph.edges()):
        global_edge_colors = get_edge_color_map(global_graph, global_min, global_max, attribute='attr')
        nx.draw_networkx_edges(global_graph, pos=pos_global, ax=ax2,
                               edge_color=[global_edge_colors[edge] for edge in global_graph.edges()])
    else:
        nx.draw_networkx_edges(global_graph, pos=pos_global, ax=ax2, edge_color='gray')

    ax2.set_title('Graph', fontsize=24)

    # 映射线
    for pair in mapping.split(':'):
        if pair:
            global_node, pattern_node = map(int, pair.split(','))
            con = ConnectionPatch(xyA=pos_global[global_node], xyB=pos_pattern[pattern_node],
                                  coordsA="data", coordsB="data",
                                  axesA=ax2, axesB=ax1, color="green")
            ax2.add_artist(con)

    plt.show()


def filter_duplicates(match_results):
    unique_matches = []
    seen_sets = set()

    for match in match_results:
        # 提取全局图中的节点ID
        global_ids = {pair.split(',')[0] for pair in match.split(':') if pair}

        # 如果这组ID之前没有遇到过，就添加到结果中
        if frozenset(global_ids) not in seen_sets:
            unique_matches.append(match)
            seen_sets.add(frozenset(global_ids))

    return unique_matches


def group_by_id_sets(match_results):
    grouped_matches = {}

    for match in match_results:
        # 提取全局图中的节点ID
        global_ids = frozenset(pair.split(',')[0] for pair in match.split(':') if pair)

        # 如果这组ID之前没有遇到过，初始化一个新列表
        if global_ids not in grouped_matches:
            grouped_matches[global_ids] = []

        # 将当前映射添加到对应的列表中
        grouped_matches[global_ids].append(match)

    # 将所有分组的列表收集到一个二维列表中
    grouped_list = list(grouped_matches.values())

    return grouped_list


def json_to_grf(json_data, is_directed=True):
    # 初始化GRF数据字符串
    grf_data = ""

    # 获取节点数量并添加到GRF数据字符串中
    num_nodes = len(json_data['node'])
    grf_data += f"{num_nodes}\n"

    # 添加每个节点的类型
    for i, node in enumerate(json_data['node']):
        node_type = json_data['node_type'][i] if 'node_type' in json_data and json_data['node_type'][
            i] != 'None' else ''
        node_type_int = NODE_TYPE_TO_INT.get(node_type, 0)  # 默认值为0，如果类型未找到
        grf_data += f"{i} {node_type_int}\n"

    # 维护一个集合，用于存储已添加的边
    added_edges = set()

    for src in range(num_nodes):
        edges_from_node = []
        for edge_index, src_node in enumerate(json_data['edge_index'][0]):
            if src_node == src:
                dst = json_data['edge_index'][1][edge_index]
                # 对于无向图，检查边是否已添加
                if not is_directed and (dst, src) in added_edges:
                    continue
                edges_from_node.append(edge_index)

        # 更新边的数量
        num_edges = len(edges_from_node)
        grf_data += f"{num_edges}\n"
        for edge_index in edges_from_node:
            dst = json_data['edge_index'][1][edge_index]
            edge_attr = json_data['edge_attr'][edge_index] if edge_index < len(json_data['edge_attr']) else ''
            edge_attr_int = EDGE_TYPE_TO_INT.get(edge_attr, 0)  # 如果类型未找到，默认值为0
            # 添加边到GRF数据字符串
            grf_data += f"{src} {dst} {edge_attr_int}\n"
            # 将边添加到已添加边的集合中
            added_edges.add((src, dst))

    return grf_data


def json_to_grf_model(json_data):
    # 初始化GRF数据字符串
    grf_data = ""

    # 获取节点数量并添加到GRF数据字符串中
    nodes = json_data['nodes']
    num_nodes = len(nodes)
    grf_data += f"{num_nodes}\n"

    # 创建节点ID到整数的映射
    id_to_int = {node['data']['id']: i for i, node in enumerate(nodes)}

    # 添加每个节点的类型
    for node in nodes:
        node_type = node['data']['type']
        node_type_int = NODE_TYPE_TO_INT.get(node_type, 0)  # 默认值为0，如果类型未找到
        grf_data += f"{id_to_int[node['data']['id']]} {node_type_int}\n"

    # 遍历每个节点，添加边
    edges = json_data['edges']
    for src_id, group in groupby(sorted(edges, key=lambda x: x['data']['source']), key=lambda x: x['data']['source']):
        group_list = list(group)
        num_edges = len(group_list)
        src_int = id_to_int[src_id]
        grf_data += f"{num_edges}\n"
        for edge in group_list:
            dst_int = id_to_int[edge['data']['target']]
            edge_type = edge['data']['type']
            edge_type_int = EDGE_TYPE_TO_INT.get(edge_type, 0)  # 默认值为0，如果类型未找到
            grf_data += f"{src_int} {dst_int} {edge_type_int}\n"

    return grf_data


def dict_to_gml(json_data, is_directed=True):
    gml_data = "graph [\n"
    gml_data += f"  directed {int(is_directed)}\n"

    # Adding nodes
    for i, node in enumerate(json_data['node']):
        gml_data += f"  node [\n    id {i}\n    label \"{node}\"\n"
        if json_data['node_type'][i] != 'None':
            gml_data += f"    type \"{json_data['node_type'][i]}\"\n"
        if json_data['node_attr'][i] != 'None':
            gml_data += f"    attribute \"{json_data['node_attr'][i]}\"\n"
        gml_data += "  ]\n"

    # Adding edges
    for i in range(len(json_data['edge_index'][0])):
        src = json_data['edge_index'][0][i]
        dst = json_data['edge_index'][1][i]
        attribute = json_data['edge_attr'][i] if i < len(json_data['edge_attr']) else 'None'

        # If the graph is directed, add all edges
        # If it is undirected, add the edge only if it doesn't end with '_R' and source < target
        if is_directed or (not attribute.endswith("_R") and src < dst):
            gml_data += f"  edge [\n    source {src}\n    target {dst}\n"
            gml_data += f"    attribute \"{attribute}\"\n"
            gml_data += "  ]\n"

    gml_data += "]"

    return gml_data


def draw_graph_from_gml(gml_data):
    # 创建图
    G = nx.parse_gml(gml_data)

    # 为不同种类的边和节点指定颜色
    edge_colors = {
        "Connected": "black",
        "Equal": "gray",
        "Perpendicular": "blue",
        "Parallel": "cyan",
        "Congruent": "magenta",
        "Similar": "lime",
        "LiesOnLine": "orange",
        "LiesOnLine_R": "brown",
        "LiesOnCircle": "purple",
        "LiesOnCircle_R": "pink",
        "Center": "gold",
        "Center_R": "olive",
        "Interior": "teal",
        "Interior_R": "navy",
        "Endpoint": "maroon",
        "Endpoint_R": "green",
        "Sidepoint": "yellow",
        "Sidepoint_R": "turquoise",
        "Side": "violet",
        "Side_R": "salmon",
        "Vertex": "red",
        "Vertex_R": "darkgreen"
    }

    node_colors = {
        "Point": "red",
        "Triangle": "green",
        "Circle": "blue",
        "Arc": "purple",
        "Polygon": "orange",
        "Line": "grey",
        "Angle": "cyan"
    }

    # 使用spring布局
    pos = nx.spring_layout(G, k=0.25, iterations=20)

    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_color = edge_colors.get(edge[2].get('attribute', 'black'), 'black')

        middlex = (x0 + x1) / 2
        middley = (y0 + y1) / 2

        # 将边的属性转换为字符串
        edge_attr = ', '.join([f"{key}: {value}" for key, value in edge[2].items()])

        # 创建带有曲线的边
        edge_trace.append(go.Scatter(x=[x0, middlex, x1, None],
                                     y=[y0, middley, y1, None],
                                     mode='lines',
                                     line=dict(width=1, color=edge_color),
                                     hoverinfo='text',
                                     text=edge_attr))  # 显示边的属性

    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                            hoverinfo="text", marker={'size': 10, 'color': []})

    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_color = node_colors.get(node[1].get('type', 'black'), 'black')
        node_trace['marker']['color'] += tuple([node_color])
        node_info = f"{node[0]}: {node[1]}"  # 显示节点属性
        node_trace['hovertext'] += tuple([node_info])
        node_trace['text'] += tuple([str(node[0])])

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='<br>Network graph with Plotly',
                        showlegend=False, hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    fig.show()
