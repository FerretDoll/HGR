from string import Template

import networkx as nx
from reasoner.config import NODE_TYPE_TO_INT, EDGE_TYPE_TO_INT


class Hologram:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_types = set()
        self.edge_types = set()
        self.grf_data = ""
        self.grf_to_id = {}

    def add_node(self, node_id, node_type=None, node_value=None, **node_attrs):
        """Add a node and its attributes
        :param node_id: Unique identifier of the node
        :param node_type: Type of node
        :param node_value: node value
        :param node_attrs: Other attributes of the node
        """
        node_attrs['type'] = node_type
        node_attrs['value'] = node_value
        self.graph.add_node(node_id, **node_attrs)

    def add_edge(self, from_node, to_node, edge_type=None, **edge_attrs):
        """Add an edge and its attributes
        :param from_node: Starting node of edge
        :param to_node: End node of edge
        :param edge_type: Type of edge
        :param edge_attrs: Other attributes of the edge
        """
        edge_attrs['type'] = edge_type
        self.graph.add_edge(from_node, to_node, **edge_attrs)

    def get_node_value(self, node_id):
        """Get the value of the node"""
        return self.graph.nodes[node_id].get('value')

    def get_node_visual_value(self, node_id):
        """Get the visual value of the node"""
        return self.graph.nodes[node_id].get('node_visual_value')

    def get_node_domain(self, node_id):
        """Get the domain of node"""
        return self.graph.nodes[node_id].get('node_domain')

    def modify_node_value(self, node_id, new_value):
        """Modify the value attribute of a node"""
        if node_id in self.graph.nodes:
            self.graph.nodes[node_id]['value'] = new_value

    def get_node_type(self, node_id):
        """Get the type of node"""
        return self.graph.nodes[node_id].get('type')

    def get_node_types(self):
        """Get the types of all nodes and count the number of each type"""
        # Initialize an empty dict to store node types and their counts
        node_types = {}

        # 遍历图中的所有节点
        for node, data in self.graph.nodes(data=True):
            # Extract type attribute
            node_type = data.get('type')  # Use the get method to avoid KeyError
            if node_type:  # Ensure that node_date is not None
                # Convert node_type to an integer, if undefined, defaults to 0
                int_type = str(NODE_TYPE_TO_INT.get(node_type, 0))
                # Update the counts in the dict
                if int_type in node_types:
                    node_types[int_type] += 1
                else:
                    node_types[int_type] = 1

        return node_types

    def get_edge_types(self):
        """Get the types of all edges and count the number of each type"""
        # Initialize an empty dict to store edge types and their counts
        edge_types = {}

        # Traverse all edges in the graph
        for node1, node2, data in self.graph.edges(data=True):
            # Extract type attribute
            edge_type = data.get('type')
            if edge_type:
                # Convert edge_type to an integer, default to 0 if undefined
                int_type = str(EDGE_TYPE_TO_INT.get(edge_type, 0))
                # Update the counts in the dict
                if int_type in edge_types:
                    edge_types[int_type] += 1
                else:
                    edge_types[int_type] = 1

        return edge_types

    def get_edge_type(self, from_node, to_node):
        """Obtain the type of edge"""
        # In networkx, the attributes of edges can be accessed through the edges attribute of graph objects,
        # and then the nodes of the edges can be used as keys to access them
        return self.graph.edges[from_node, to_node].get('type')

    def update_grf_to_id(self):
        """Update the mapping function from GRF ID to networkx ID"""
        # Clear the current grf_to_id dict
        self.grf_to_id.clear()
        # Rebuilding grf_to_id dict
        for i, node_id in enumerate(self.graph.nodes):
            self.grf_to_id[i] = node_id


class GlobalHologram(Hologram):
    def __init__(self):
        super().__init__()
        self.target = None
        self.target_equation = None
        self.point_positions = None
        self.equations = None

    def update_grf_data(self):
        new_grf_data = ""

        # Obtain the number of nodes
        num_nodes = self.graph.number_of_nodes()
        new_grf_data += f"{num_nodes}\n"

        # Add the type of each node
        for i, node_id in enumerate(self.graph.nodes):
            node_type = self.graph.nodes[node_id].get('type', '')
            node_type_int = NODE_TYPE_TO_INT.get(node_type, 0)  # The default value is 0, if the type is not found
            new_grf_data += f"{i} {node_type_int}\n"

        added_edges = set()

        for src_node in self.graph.nodes:
            src_index = list(self.graph.nodes).index(src_node)  # Retrieve the index of the source node
            edges_from_node = []

            for dst_node in self.graph.adj[src_node]:
                dst_index = list(self.graph.nodes).index(dst_node)  # Retrieve the index of the target node

                # For undirected graphs, check if edges have been added
                if (dst_index, src_index) in added_edges:
                    continue

                edges_from_node.append(dst_node)

            # Update the number of edges
            num_edges = len(edges_from_node)
            new_grf_data += f"{num_edges}\n"

            for dst_node in edges_from_node:
                dst_index = list(self.graph.nodes).index(dst_node)  # Retrieve the index of the target node
                edge_attr = self.graph.edges[src_node, dst_node].get('type', '')
                edge_attr_int = EDGE_TYPE_TO_INT.get(edge_attr, 0)  # If the type is not found, the default value is 0

                # Add edges to GRF data string
                new_grf_data += f"{src_index} {dst_index} {edge_attr_int}\n"

                # Add edges to the collection of added edges
                added_edges.add((src_index, dst_index))

        self.grf_data = new_grf_data

    @classmethod
    def from_dict(cls, json_data):
        """Instantiate GlobalGraph based on JSON data"""
        global_graph = cls()

        id_map = {}
        # Traverse nodes and add them to the graph
        for i, node_name in enumerate(json_data['node']):
            node_type = json_data['node_type'][i] if (
                    'node_type' in json_data and json_data['node_type'][i] != 'None') else ''
            node_value = json_data['node_attr'][i] if 'node_attr' in json_data else {}
            node_visual_value = json_data['node_visual_attr'][i] if 'node_visual_attr' in json_data else {}
            node_domain = json_data['node_domain'][i] if 'node_domain' in json_data else {}
            # Add nodes to the graph
            global_graph.add_node(node_name, node_type=node_type, node_value=node_value,
                                  node_visual_value=node_visual_value, node_domain=node_domain)
            id_map[i] = node_name

        # Maintain a collection for storing added edges, ensuring that edges in the undirected graph are not duplicated
        added_edges = set()

        # Traverse the edge index and add edges to the graph
        for edge_index, src_node in enumerate(json_data['edge_index'][0]):
            dst_node = json_data['edge_index'][1][edge_index]
            # For undirected graphs, check if edges have been added
            if (dst_node, src_node) in added_edges:
                continue

            # Read edge attributes, if present
            edge_type = json_data['edge_attr'][edge_index] if edge_index < len(json_data['edge_attr']) else {}
            # Add edges to the graph
            global_graph.add_edge(id_map[src_node], id_map[dst_node], edge_type=edge_type)

            # Add edges to the collection of added edges
            added_edges.add((src_node, dst_node))

        # Set problem target node
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
        """Convert the global graph to dict format"""
        # Extract node information
        nodes = list(self.graph.nodes)
        node_type = [self.get_node_type(node) for node in nodes]
        node_attr = [self.get_node_value(node) for node in nodes]
        target_node = self.target

        # Extract edge information
        edge_index = [list(edge) for edge in zip(*self.graph.edges)]
        st_idx = [nodes.index(st) for st in edge_index[0]]
        ed_idx = [nodes.index(ed) for ed in edge_index[1]]
        new_edge_index = [st_idx, ed_idx]
        edge_attr = [self.get_edge_type(edge[0], edge[1]) for edge in self.graph.edges]

        # Find all nodes of the same type and compare them
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

        # Add Equal edges to edge information
        for (st, ed) in equal_edges:
            new_edge_index[0].append(st)
            new_edge_index[1].append(ed)
            edge_attr.append('Equal')

        # Building a dict
        graph_dict = {
            "node": nodes,
            "node_type": node_type,
            "node_attr": node_attr,
            "edge_index": new_edge_index,
            "edge_attr": edge_attr,
            "target_node": target_node
        }

        return graph_dict


class GraphModel(Hologram):
    def __init__(self):
        super().__init__()
        self.model_id = ""
        self.model_name = ""
        self.relation_template = ""
        self.constraints = ""
        self.visual_constraints = ""
        self.actions = []
        self.equations = []
        self.fixed_nodes = []

    def generate_relation(self, mapping_dict):
        """
        Generate transformed relation strings based on relation template and mapping function.

        :param mapping_dict: Mapping dictionary, where keys are placeholders in the template
        and values are corresponding node names or IDs.
        :return: Replaced the relationship string of placeholders with values from the mapping dictionary.
        """
        template = Template(self.relation_template)  # Create Template Object
        relation_str = template.substitute(mapping_dict)  # Replace placeholders in templates with mapping dictionaries
        return relation_str

    @classmethod
    def from_json(cls, json_data, model_name):
        """Instantiate GraphModel based on JSON data"""
        model_graph = cls()

        # Directly parse the incoming JSON data
        model_id = json_data.get("id", "")
        nodes = json_data.get("nodes", [])
        edges = json_data.get("edges", [])
        graph = json_data.get("grf_data", "")
        relation = json_data.get("relation", "")
        constraints = json_data.get("constraints", "")
        visual_constraints = json_data.get("visual_constraints", "")
        fixed_nodes = json_data.get("fixed_nodes", [])
        actions = json_data.get("actions", [])
        equations = json_data.get("equations", [])
        node_types = json_data.get("node_types", "")
        edge_types = json_data.get("edge_types", "")

        # Assign values to GraphModel instance
        model_graph.model_id = model_id
        model_graph.model_name = model_name  # Set Model ID
        model_graph.grf_data = graph  # Set the representation data for the graph
        model_graph.relation_template = relation  # set relation
        model_graph.constraints = constraints  # Set mathematical constraints
        model_graph.visual_constraints = visual_constraints  # Set visual constraints
        model_graph.fixed_nodes = fixed_nodes
        model_graph.actions = actions  # Set action
        model_graph.equations = equations  # Set equation
        model_graph.node_types = node_types
        model_graph.edge_types = edge_types

        # Firstly, handle the nodes and establish a mapping from ID to GRF
        id_map = {}
        for node in nodes:
            node_name = node["name"]  # The name in the JSON file is used as the node_name
            node_type = node["type"]

            # Add nodes to GraphModel
            model_graph.add_node(node_name, node_type=node_type)

            # Establish a mapping from ID to name for subsequent edge processing
            id_map[node["id"]] = node_name

        # Process the edges and convert the source and target from their original IDs to the corresponding new node_id
        for edge in edges:
            source_id = id_map.get(edge["source"])  # Get the new node_id of the edge starting point
            target_id = id_map.get(edge["target"])  # Get the new node_id of the edge endpoint
            edge_type = edge["type"]

            if source_id is not None and target_id is not None:
                # Add edges to GraphModel and use the new node_id as a reference
                model_graph.add_edge(source_id, target_id, edge_type=edge_type)

        model_graph.update_grf_to_id()

        return model_graph
