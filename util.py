import pandas as pd 
import re 
import networkx as nx 
from datetime import date 

## GLOBALS VARIABLES

patterns = {
    "quoted_name": '(\"([^,])*,([^\"])*\")',
    "parentheses": '(\(([^@])*@([^\]])*\))',
    "square_brackets": '(\[([^@])*@([^\]])*\])',
    "angle_brackets": '(<([^@])*@([^>])*>)',
    "multiple_semicolons": ';{2,}'
}

email_regex = "(([a-zA-Z\.\d\-])*@([a-zA-Z\.\d\-])*)"

## DATA CLEANING FUNCTIONS ##
def clean_mixed_format_emails(emails:str) ->str:
    
    if '"' not in emails and ';' in emails:
        return emails 
    
    # finds names in quotation marks that have either a comma or semicolon
    matches = re.findall('(\"([^,])*[,;]([^\"])*\")', emails)
    names = [m[0] for m in matches]
    for n in names:
        emails = emails.replace(n,'')
    emails = emails.replace(',,', ',').replace(',', ';').strip()
    
    if len(names) > 0:
        names = ';'.join(names)
        if emails[-1:] == ';':
            emails = emails + names
        elif len(emails) > 0 and emails[-1:] != ";":
            emails = emails + ';' + names
        elif len(emails) == 0:
            emails = names 
    if emails[0:1] == ";":
        emails = emails[1:]
    
    matches = re.findall(patterns['multiple_semicolons'], emails)
    for m in matches:
        emails = emails.replace(m, ';')
        
    return emails 


def split_recipients(recipients:str, separator = ';'):
    r_list = []
    if len(recipients) > 0:
        r_list = [ extract_emails(r)[0].strip() for r in recipients.split(separator) ]
    return r_list
    

def extract_emails(text:str) ->list:
    text = text.lower()
    matches = re.findall(email_regex, text)
    emails = [ m[0] for m in matches ]
    if len(emails) == 0:
        emails = [text]
    return emails


def clean_columns(df:pd.DataFrame, fixes:dict = {})->pd.DataFrame:
    """
    renames columns with an optional dictionary of column names
    Args:
        df (pd.DataFrame): dataframe to clean column names
        fixes (dict, optional): dictionary of columns to rename
    Returns:
        pd.DataFrame: dataframe with renamed / converted columns
    """
    lowercase = { 
        c: c.lower().strip().replace(' ', '_').replace('\n', '_') 
        for c in df.columns }
    df = df.rename(columns=lowercase)
    df = df.rename(
            columns = {f: fixes[f] for f in fixes if f in df.columns}
    )
    return df

def reformat_log(df: pd.DataFrame, fields:dict, filename:str, full_emails_only = False):
    print("reformatting log")
    records = []
    recipient_map = {}
    for row in df.to_dict('records'):
        sender = row.get(fields['from'])
        if isinstance(sender, str):
            msg_from = extract_emails(sender)[0] 
        else:
            msg_from = "" 
            
        msg_id = row[fields['id']]
        
        for column in fields['to']:
            recipients = row.get(column, "")
            if pd.notna(recipients):
                # convert the text of recepients into a list of recipients
                if recipients in recipient_map:
                    recipient_list = recipient_map[recipients]
                else:            
                    if full_emails_only is True:
                        recipient_list = extract_emails(recipients)
                    else:
                        recipient_list = split_recipients( clean_mixed_format_emails(recipients) )
                    recipient_map[recipients] = recipient_list
                
                # add a record for each recipient     
                for r in recipient_list:
                    records.append(
                        {
                            "filename": filename,
                            "id": msg_id,
                            "from": msg_from, 
                            "to": r, 
                            "type": column, 
                            "sender_domain": msg_from.split('@').pop() if '@' in msg_from else None, 
                            "recipient_domain": r.split('@').pop() if '@' in r else None
                        }
                    )
    rfl = pd.DataFrame(records).drop_duplicates()
    return rfl

def count_emails(df):
    return (
                df[ (pd.notna(df['from']) ) & (pd.notna(df['to'])) ]
                .fillna('')
                .groupby(['filename', 'from', 'to', 'type', 'sender_domain', 'recipient_domain'])
                .id.nunique()
                .rename("emails")
                .reset_index()
            )

### NETWORK GRAPH FUNCTIONS
def build_graph(G:nx.MultiDiGraph, graph_factory, graph_df:pd.DataFrame) -> nx.MultiDiGraph:
    print(f"Building graph from {len(graph_df)} rows")
    G = nx.compose(G, graph_factory.make_graphs([ e for e in graph_df.to_dict('records') ]))
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f"{len(G)} nodes")
    return G 
    
    
def community_colors(g):
        
    #px.colors.qualitative.Plotly
    domain_colors = [
        '#636EFA',
        '#EF553B',
        '#00CC96',
        '#AB63FA',
        '#FFA15A',
        '#19D3F3',
        '#FF6692',
        '#B6E880',
        '#FF97FF',
        '#FECB52'
    ]

    #px.colors.qualitative.Set1
    node_colors = [
        'rgb(228,26,28)',
        'rgb(55,126,184)',
        'rgb(77,175,74)',
        'rgb(152,78,163)',
        'rgb(255,127,0)',
        'rgb(255,255,51)',
        'rgb(166,86,40)',
        'rgb(247,129,191)',
        'rgb(153,153,153)'
    ]

    #px.colors.qualitative.Pastel1
    edge_colors = [
        'rgb(251,180,174)',
        'rgb(179,205,227)',
        'rgb(204,235,197)',
        'rgb(222,203,228)',
        'rgb(254,217,166)',
        'rgb(255,255,204)',
        'rgb(229,216,189)',
        'rgb(253,218,236)',
        'rgb(242,242,242)'
    ]
    
    communities = nx.algorithms.community.greedy_modularity_communities(g)

    for node_id in g.nodes:
        node = g.nodes[node_id]
        for community_counter, community_members in enumerate(communities):
            if node_id in community_members:
                break
        node['community'] = node_colors[community_counter % len(node_colors)]
        
    for edge_id in g.edges:
        edge =  g.edges[edge_id]
        source_node = g.nodes[edge_id[0]]
        target_node = g.nodes[edge_id[1]]
        pastel = edge_colors[node_colors.index(source_node['community'])]
        edge['community'] =  pastel if source_node['community'] == target_node['community'] else 'lightgray'
    
    # node_colors = dict(nx.get_node_attributes(g, "community", default='#acacac'))
    # print(node_colors)
    return g


def get_node_colors(G, color_field:str = "type"):
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
    
    node_types = dict(
        pd.Series(
            nx.get_node_attributes(G, color_field, default=None)
            .values()
        ).value_counts()
    ).keys()
    
    diff = len(node_types) - len(colors)
    if diff > 0:
        for nt in range(diff):
            colors.append("#d3d3d3")
            
    node_colors = get_colormap(node_types, colors) 
    return node_colors 

def get_colormap(values, colors):
    return { v: colors[count] for count, v in enumerate(values)} 

def merge_graphs(G1:nx.DiGraph, G2: nx.DiGraph): 
    combined = nx.compose(G1, G2)
    node_data = {}
    for n in G1.nodes & G2.nodes:
        f1 = G1.nodes[n]['files']
        f2 = G2.nodes[n]['files']
        if isinstance(f1, str):
            f1 = eval(f1)
        if isinstance(f2, str):
            f2 = eval(f2)
            
        node_data[n] = list(set(f1 + f2))
        
    # node_data = { n: list(set(G1.nodes[n]['files'] + G2.nodes[n]['files'])) for n in G1.nodes & G2.nodes}
    nx.set_node_attributes(combined, node_data, 'files')
    return combined


def combine_nodes(G, nodes:list):
    keep_node = nodes[0]
    merge_data = {}
    final_type = G.nodes[keep_node]['type']

    for n in nodes:
        if n in G.nodes and n != keep_node:                
            merge_data[n] = G.nodes[n]
            G = nx.identified_nodes(G, keep_node, n)
            
    if keep_node in G:
        G.nodes[keep_node]['alias_ids'] = nodes
        md = G.nodes[keep_node]['merge_data'] if "merge_data" in G.nodes[keep_node].keys() else {}
        G.nodes[keep_node]['merge_data'] = { **md, **merge_data}
        G.nodes[keep_node]['type'] = final_type
        
    return G 

def get_node_names(G)->dict:
    node_names = {} 
    for n in G.nodes:
        name = G.nodes[n].get("label", n)
        node_names[name] = n
    return node_names


def match_column(columns:list, search:str, exclude:list = []) -> str|None: 
    for col in columns:
        if col is not None and search in col and col not in exclude:
            return col 

        
def match_columns(columns:list, searches:list[str], exclude:list = []) -> list|None:
    matches = []
    for col in columns:
        for s in searches:
            if col is not None and s in col and col not in exclude:
                matches.append(col)
    return list(set(matches)) if len(matches) > 0 else None 
    