import io 
import msgspec
from ipysigma import Sigma
import networkx as nx 
from typing import Optional
import json     
class Element(msgspec.Struct):
    type : str 
    value : str 


class Node(msgspec.Struct):
    id : str 
    label : str 
    type : str 
    data_source : str = ""
    attr : dict = {}
    
    def nx_format(self):
        return (self.id, {"label": self.label, "type": self.type, "data_source": self.data_source, **self.attr})


class NodeFactory(msgspec.Struct, kw_only=True):
    id_field : str
    label_field : Optional[str | None] = None
    type : Optional[Element]
    attr : list[str] = []
    tidy : Optional[str|None] = None
    
    def make_node(self, data:dict, data_source:str = ""):
        return Node(**{
            "id": str(data.get(self.id_field)),
            "label": str(data.get(self.label_field)) if self.label_field else str(data.get(self.id_field)),
            "type": data.get(self.type.value) if self.type.type == "field" else self.type.value,
            "attr": {**{a: data.get(a, None) for a in self.attr}, "tidy": self.tidy, "data_source": data_source}
        })
        
    def to_dict(self):
        return {
            "id": self.id_field, 
            "label": self.label_field,
            "type": f'col:{self.type.value}' if self.type == "field" else self.type.value,
            "details": self.attr
        }


class Link(msgspec.Struct):
    source : str 
    target : str 
    type : str 
    attr : dict = {}
    
    def nx_format(self) -> tuple:
        return (str(self.source), str(self.target), {"type": self.type, **self.attr})

    
class LinkFactory(msgspec.Struct):
    source_field : str 
    target_field: str 
    type : Optional[Element]
    attr : list[str] = []
    
    def to_dict(self):
        linked_by = f'col:{self.type.value}' if self.type.type == "field" else self.type.value
        return {
            "source": self.source_field, 
            "target": self.target_field,
            "linked by": linked_by, 
            "details": self.attr
        }
        
    def type_check(self, detail):
        try:
            return float(detail)
        except Exception as e:
            return detail
        
        
    def make_link(self, data:dict):
        if data.get(self.source_field) is not None and data.get(self.target_field) is not None:
            attr = { a: self.type_check( data.get(a, None) ) for a in self.attr }
            try:
                if "details" in attr and isinstance(attr['details'], str): 
                    details = json.loads(attr['details'])
                    attr = { **attr, **details}
                    for d in details:
                        if 'amount' in d: 
                            attr['size'] = details[d]
                            
            except Exception as e:
                print(e, attr['details'])
                pass 
                
            return(Link(**{
                "source": data.get(self.source_field), 
                "target": data.get(self.target_field), 
                "type":  data.get(self.type.value) if self.type.type == "field" else self.type.value,
                "attr": attr
            }))



class GraphSchema(msgspec.Struct):
    node_factories: dict[str, NodeFactory]
    link_factories: list[LinkFactory]

    
class GraphFactory(msgspec.Struct):
    node_factories : list[NodeFactory]
    link_factories : list[LinkFactory]
    
    def make_nodes(self, data:dict, data_source:str = ""):
        return [ nf.make_node(data, data_source) for nf in self.node_factories ]
    
    def nx_nodes(self, data:dict, data_source:str = ""):
        nodes = self.make_nodes(data, data_source)
        return [ node.nx_format() for node in nodes if node is not None]
    
    def make_links(self, data:dict):
        return [ lf.make_link(data) for lf in self.link_factories ]
    
    def nx_edges(self, data:dict):
        links = self.make_links(data)
        return [ link.nx_format() for link in links if link is not None]
    
    def make_graphs(self, data:list, data_source:str = ""):
        nodes = []
        edges = []
        for d in data:
            nodes += self.nx_nodes(d, data_source)
            edges += self.nx_edges(d)
                    
        G = nx.MultiDiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G 
    
    def make_graph(self, data:dict, data_source:str):
        G = nx.MultiDiGraph()
        G.add_nodes_from(self.nx_nodes(data, data_source))
        G.add_edges_from(self.nx_edges(data))
        return G 


class SigmaFactory(msgspec.Struct):
    height : int = 1000
    layout_settings : dict | None = None
    
    edge_color : str = "type"
    edge_size : str | None = None
    edge_size_range: tuple = (1, 15)
    edge_weight : str | None = None
    edge_color_palette : str | None = "Category10"
    edge_color_gradient: tuple | None = None 
    default_edge_color : str = "gray"
    
    node_size : str | dict | None = None
    node_size_range : tuple[int, int] = (3, 30)
    node_color : str = "type"
    node_color_palette: str | dict = "Dark2"
    node_metrics : dict | None = None
    raw_node_color: str | None = None
    
    clickable_edges : bool = False
    selected_node : str|None = None
    layout : dict|None = None
    layout_settings : dict|None
    camera_state : dict = {}
    
    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}
        
    def make_sigma(self, G:nx.MultiGraph, **kwargs):
        
        settings = { f: kwargs.get(f, getattr(self, f)) for f in self.__struct_fields__}
        settings['raw_node_border_color'] = "white"
        settings['raw_node_border_size'] = "1px"
        settings['raw_node_border_ratio'] = ".01"
        settings['start_layout'] = len(G) / 20
        settings = {key: settings[key] for key in settings if settings[key] is not None}
        
        return Sigma(
            G, 
            **settings
        )
        
    
    def export_graph(self, G:nx.MultiGraph, **kwargs):
        with io.BytesIO() as bytes_buf:
            with io.TextIOWrapper(bytes_buf) as text_buf:
                settings = { f: kwargs.get(f, getattr(self, f)) for f in self.__struct_fields__}
                settings['raw_node_border_color'] = "white"
                settings['raw_node_border_size'] = "1px"
                settings['raw_node_border_ratio'] = ".01"
                settings['start_layout'] = len(G) / 20
                settings['path'] = text_buf
                
                Sigma.write_html( G,**settings)
                yield bytes_buf.getvalue()
                
                
class QNG(msgspec.Struct):
    adjacency: dict
    node_attrs: dict
    sigma_factory: SigmaFactory
    def multigraph(self):
        MG = nx.from_dict_of_dicts(self.adjacency, multigraph_input=True, create_using=nx.MultiDiGraph)
        nx.set_node_attributes(MG, self.node_attrs)
        return MG 
    
     
    
def load_schema(filename:str):
    with open(filename, 'r') as f:
        return msgspec.json.decode(f.read(), type=GraphSchema)