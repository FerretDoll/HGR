from string import Template

import networkx as nx
from reasoner.config import NODE_TYPE_TO_INT, EDGE_TYPE_TO_INT


class LogicGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_types = set()
        self.edge_types = set()
        self.grf_data = ""
        self.grf_to_id = {}

    def add_node(self, node_id, node_type=None, node_value=None, **node_attrs):
        """添加一个节点及其属性
        :param node_id: 节点的唯一标识
        :param node_type: 节点的类型
        :param node_value: 节点的值
        :param node_attrs: 节点的其他属性
        """
        node_attrs['type'] = node_type
        node_attrs['value'] = node_value
        self.graph.add_node(node_id, **node_attrs)

    def add_edge(self, from_node, to_node, edge_type=None, **edge_attrs):
        """添加一条边及其属性
        :param from_node: 边的起始节点
        :param to_node: 边的结束节点
        :param edge_type: 边的类型
        :param edge_attrs: 边的其他属性
        """
        edge_attrs['type'] = edge_type  # 添加边的类型到边属性中
        self.graph.add_edge(from_node, to_node, **edge_attrs)

    def get_node_value(self, node_id):
        """获取节点的value"""
        return self.graph.nodes[node_id].get('value')

    def get_node_visual_value(self, node_id):
        """获取节点的visual_value"""
        return self.graph.nodes[node_id].get('node_visual_value')

    def modify_node_value(self, node_id, new_value):
        """修改节点的value属性"""
        if node_id in self.graph.nodes:
            self.graph.nodes[node_id]['value'] = new_value

    def get_node_type(self, node_id):
        """获取节点的type"""
        return self.graph.nodes[node_id].get('type')

    def get_node_types(self):
        """获取所有节点的type，并统计每种类型的数量"""
        # 初始化一个空字典来存储节点类型及其计数
        node_types = {}

        # 遍历图中的所有节点
        for node, data in self.graph.nodes(data=True):
            # 提取type属性
            node_type = data.get('type')  # 使用get方法以避免KeyError
            if node_type:  # 确保node_type不是None
                # 将node_type转换为整数，如果未定义则默认为0
                int_type = str(NODE_TYPE_TO_INT.get(node_type, 0))
                # 更新字典中的计数
                if int_type in node_types:
                    node_types[int_type] += 1
                else:
                    node_types[int_type] = 1

        return node_types

    def get_edge_types(self):
        """获取所有边的type，并统计每种类型的数量"""
        # 初始化一个空字典来存储边类型及其计数
        edge_types = {}

        # 遍历图中的所有边
        for node1, node2, data in self.graph.edges(data=True):
            # 提取type属性
            edge_type = data.get('type')
            if edge_type:
                # 将edge_type转换为整数，如果未定义则默认为0
                int_type = str(EDGE_TYPE_TO_INT.get(edge_type, 0))
                # 更新字典中的计数
                if int_type in edge_types:
                    edge_types[int_type] += 1
                else:
                    edge_types[int_type] = 1

        return edge_types

    def get_edge_type(self, from_node, to_node):
        """获取边的type"""
        # 在networkx中，边的属性可以通过graph对象的edges属性，然后使用边的节点作为键来访问
        return self.graph.edges[from_node, to_node].get('type')

    def update_grf_to_id(self):
        """更新 GRF ID 到 networkx ID 的映射字典"""
        # 清空当前的name_to_id字典
        self.grf_to_id.clear()
        # 重新构建id_to_grf字典
        for i, node_id in enumerate(self.graph.nodes):
            self.grf_to_id[i] = node_id


class GlobalGraph(LogicGraph):
    def __init__(self):
        super().__init__()
        self.target = None
        self.target_equation = None
        self.point_positions = None
        self.equations = None

    def update_grf_data(self):
        new_grf_data = ""

        # 获取节点数量
        num_nodes = self.graph.number_of_nodes()
        new_grf_data += f"{num_nodes}\n"

        # 添加每个节点的类型
        for i, node_id in enumerate(self.graph.nodes):
            node_type = self.graph.nodes[node_id].get('type', '')
            node_type_int = NODE_TYPE_TO_INT.get(node_type, 0)  # 默认值为0，如果类型未找到
            new_grf_data += f"{i} {node_type_int}\n"

        added_edges = set()

        for src_node in self.graph.nodes:
            src_index = list(self.graph.nodes).index(src_node)  # 获取源节点索引
            edges_from_node = []

            for dst_node in self.graph.adj[src_node]:
                dst_index = list(self.graph.nodes).index(dst_node)  # 获取目标节点索引

                # 对于无向图，检查边是否已添加
                if (dst_index, src_index) in added_edges:
                    continue

                edges_from_node.append(dst_node)

            # 更新边的数量
            num_edges = len(edges_from_node)
            new_grf_data += f"{num_edges}\n"

            for dst_node in edges_from_node:
                dst_index = list(self.graph.nodes).index(dst_node)  # 获取目标节点索引
                edge_attr = self.graph.edges[src_node, dst_node].get('type', '')
                edge_attr_int = EDGE_TYPE_TO_INT.get(edge_attr, 0)  # 如果类型未找到，默认值为0

                # 添加边到GRF数据字符串
                new_grf_data += f"{src_index} {dst_index} {edge_attr_int}\n"

                # 将边添加到已添加边的集合中
                added_edges.add((src_index, dst_index))

        self.grf_data = new_grf_data

    @classmethod
    def from_dict(cls, json_data):
        """根据JSON数据实例化GlobalGraph"""
        global_graph = cls()

        id_map = {}
        # 遍历节点，添加到图中
        for i, node_name in enumerate(json_data['node']):
            node_type = json_data['node_type'][i] if (
                    'node_type' in json_data and json_data['node_type'][i] != 'None') else ''
            node_value = json_data['node_attr'][i] if 'node_attr' in json_data else {}
            node_visual_value = json_data['node_visual_attr'][i] if 'node_visual_attr' in json_data else {}
            # 添加节点到图中
            global_graph.add_node(node_name, node_type=node_type, node_value=node_value,
                                  node_visual_value=node_visual_value)
            id_map[i] = node_name

        # 维护一个集合，用于存储已添加的边，确保无向图中边不重复
        added_edges = set()

        # 遍历边索引，添加边到图中
        for edge_index, src_node in enumerate(json_data['edge_index'][0]):
            dst_node = json_data['edge_index'][1][edge_index]
            # 对于无向图，检查边是否已添加
            if (dst_node, src_node) in added_edges:
                continue

            # 读取边属性，如果存在
            edge_type = json_data['edge_attr'][edge_index] if edge_index < len(json_data['edge_attr']) else {}
            # 添加边到图中
            global_graph.add_edge(id_map[src_node], id_map[dst_node], edge_type=edge_type)

            # 将边添加到已添加边的集合中
            added_edges.add((src_node, dst_node))

        # 设置目标节点
        global_graph.target = json_data.get("target_node")
        global_graph.target_equation = json_data.get("target_equation")
        global_graph.point_positions = json_data.get("point_positions")
        global_graph.equations = json_data.get("equations")

        global_graph.node_types = global_graph.get_node_types()
        global_graph.edge_types = global_graph.get_edge_types()
        global_graph.update_grf_data()
        global_graph.update_grf_to_id()

        return global_graph

    def to_dict(self):
        """将全局图转换为dict格式"""
        # 提取节点信息
        nodes = list(self.graph.nodes)
        node_type = [self.get_node_type(node) for node in nodes]
        node_attr = [self.get_node_value(node) for node in nodes]
        target_node = self.target

        # 提取边信息
        edge_index = [list(edge) for edge in zip(*self.graph.edges)]
        st_idx = [nodes.index(st) for st in edge_index[0]]
        ed_idx = [nodes.index(ed) for ed in edge_index[1]]
        new_edge_index = [st_idx, ed_idx]
        edge_attr = [self.get_edge_type(edge[0], edge[1]) for edge in self.graph.edges]

        # 找到所有类型相同的节点并进行比较
        type_to_nodes = {}
        for node, n_type, n_attr in zip(nodes, node_type, node_attr):
            if n_type == 'Point':
                continue
            if n_type not in type_to_nodes:
                type_to_nodes[n_type] = []
            type_to_nodes[n_type].append((node, n_attr))

        equal_edges = []
        for n_type, node_list in type_to_nodes.items():
            for i in range(len(node_list) - 1):
                node_i, attr_i = node_list[i]
                for j in range(i + 1, len(node_list)):
                    node_j, attr_j = node_list[j]
                    if attr_i == 'None' and attr_j == 'None':
                        continue
                    if attr_i != 'None' and attr_j != 'None':
                        if set(attr_i).intersection(set(attr_j)):
                            equal_edges.append((nodes.index(node_i), nodes.index(node_j)))
                        else:
                            if any(x == node_j for x in attr_i) or any(y == node_i for y in attr_j):
                                equal_edges.append((nodes.index(node_i), nodes.index(node_j)))
                    else:
                        if attr_i == 'None' and any(x == node_j for x in attr_j):
                            equal_edges.append((nodes.index(node_i), nodes.index(node_j)))
                        elif attr_j == 'None' and any(x == node_i for x in attr_i):
                            equal_edges.append((nodes.index(node_i), nodes.index(node_j)))

        # 将Equal边添加到边信息中
        for (st, ed) in equal_edges:
            new_edge_index[0].append(st)
            new_edge_index[1].append(ed)
            edge_attr.append('Equal')

        # 构建字典
        graph_dict = {
            "node": nodes,
            "node_type": node_type,
            "node_attr": node_attr,
            "edge_index": new_edge_index,
            "edge_attr": edge_attr,
            "target_node": target_node
        }

        return graph_dict


class ModelGraph(LogicGraph):
    def __init__(self):
        super().__init__()
        self.model_id = ""
        self.model_name = ""
        self.relation_template = ""
        self.constraints = ""
        self.visual_constraints = ""
        self.actions = []
        self.equations = []

    def generate_relation(self, mapping_dict):
        """
        根据关系模板和映射字典生成转换后的关系字符串。

        :param mapping_dict: 映射字典，键是模板中的占位符，值是对应的节点名称或ID。
        :return: 使用映射字典中的值替换了占位符的关系字符串。
        """
        template = Template(self.relation_template)  # 创建Template对象
        relation_str = template.substitute(mapping_dict)  # 使用映射字典替换模板中的占位符
        return relation_str

    @classmethod
    def from_json(cls, json_data, model_name):
        """根据JSON数据实例化ModelGraph"""
        model_graph = cls()

        # 直接解析传入的json_data，这里假设json_data已经是"Triangle"键下的数据
        model_id = json_data.get("id", "")
        nodes = json_data.get("nodes", [])
        edges = json_data.get("edges", [])
        graph = json_data.get("grf_data", "")
        relation = json_data.get("relation", "")
        constraints = json_data.get("constraints", "")
        visual_constraints = json_data.get("visual_constraints", "")
        actions = json_data.get("actions", [])
        equations = json_data.get("equations", [])
        node_types = json_data.get("node_types", "")
        edge_types = json_data.get("edge_types", "")

        # 为ModelGraph实例赋值
        model_graph.model_id = model_id
        model_graph.model_name = model_name  # 设置模型ID
        model_graph.grf_data = graph  # 设置图的表示数据
        model_graph.relation_template = relation  # 设置关系
        model_graph.constraints = constraints  # 设置约束条件
        model_graph.visual_constraints = visual_constraints  # 设置视觉约束条件
        model_graph.actions = actions  # 设置动作
        model_graph.equations = equations  # 设置方程
        model_graph.node_types = node_types
        model_graph.edge_types = edge_types

        # 首先处理节点，建立id到grf的映射
        id_map = {}
        for node in nodes:
            node_name = node["name"]  # json文件中的name作为node_id
            node_type = node["type"]

            # 添加节点到ModelGraph
            model_graph.add_node(node_name, node_type=node_type)

            # 建立id到name的映射，供后续处理边时使用
            id_map[node["id"]] = node_name

        # 处理边，将source和target由原始id转换为对应的新node_id
        for edge in edges:
            source_id = id_map.get(edge["source"])  # 获取边起点的新node_id
            target_id = id_map.get(edge["target"])  # 获取边终点的新node_id
            edge_type = edge["type"]

            if source_id is not None and target_id is not None:
                # 添加边到ModelGraph，使用新的node_id作为引用
                model_graph.add_edge(source_id, target_id, edge_type=edge_type)

        model_graph.update_grf_to_id()

        return model_graph
