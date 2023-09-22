from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget
from shiny.types import FileInfo
from htmltools import TagList, div
from datetime import date 
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from networkx.classes import filters
import re 
import io
from uuid import uuid4
from ipysigma import Sigma
import numpy as np 

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
def clean_mixed_format_emails(emails:str):
    
    if pd.isna(emails):
        return emails
    
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
    if pd.notna(recipients):
        recipient_list = recipients.split(separator)
        return [ extract_emails(r) for r in recipient_list ]
    else: 
        return float('nan') 
    

def parse_emails(df:pd.DataFrame, fields:dict, full_emails_only = True):
    for f in fields:
        if fields[f] is not None:
            df[f] = df[fields[f]]

    df['from'] = df['from'].apply(lambda x: extract_emails(x)[0] if pd.notna(x) else x)

    single_edges = pd.DataFrame()
    for f in ['to', 'cc']:
        if fields[f] is not None:
            single_edges = pd.concat([single_edges, clean_recipients(df, f, full_emails_only)])

    df = single_edges.copy()        
    df['sender_domain'] = df['from'].apply(lambda x: extract_domain(x))        
    return df.drop_duplicates().pipe(lambda df: df[df['from'].notna()])


def clean_recipients(df:pd.DataFrame, f:str, full_emails_only=True):
    
    clean_map = get_clean_map(df, f, full_emails_only)
    rows = []
    for row in df.to_dict('records'):
        row['recipients'] =  clean_map.get(row[f], row[f])
        row["relationship"] = f
        rows.append(row)
            
    keep_cols = ['from', 'relationship', 'recipients',  'email_id', 'date']
    cleaned = pd.DataFrame(rows).explode('recipients').reset_index()
    
    return cleaned[[c for c in keep_cols if c in cleaned.columns]]


def get_clean_map(df:pd.DataFrame, account_field:str, full_emails_only=True):
    
    accounts = list(df[account_field].dropna().unique())
    account_map = {}
    
    for account in accounts:
        if full_emails_only is False:
            account_map[account] =  split_recipients(clean_mixed_format_emails(account))
            
        elif full_emails_only is True:
            account_map[account] = extract_emails(account)
    
    if account_field == 'from':
        account_map = {a : account_map[a][0] for a in account_map}
        
    return account_map 


def extract_emails(text:str):
    if pd.isna(text):
        emails = float('nan')
    else: 
        text = text.lower()
        matches = re.findall(email_regex, text)
        emails = [ m[0] for m in matches ]
    if len(emails) == 0:
        emails = [text]
    return emails


def clean_columns(df:pd.DataFrame, fixes:dict = {})->pd.DataFrame:
    """
    converts column names to snake_case  
    renames columns with an optional dictionary of column names
    applies convert_dtypes
    typically applied with  .pipe(clean_columns )

    Args:
        df (pd.DataFrame): dataframe to clean column names
        fixes (dict, optional): dictionary of columns to rename
    Returns:
        pd.DataFrame: dataframe with renamed / converted columns
    """
    # df = df.astype('str')
    lowercase = { 
        c: c.lower().strip().replace(' ', '_').replace('\n', '_') 
        for c in df.columns }
    df = df.rename(columns=lowercase)
    df = df.rename(
            columns = {f: fixes[f] for f in fixes if f in df.columns}
    )
    return df


def extract_domain(email:str) -> str:
    domain = float('nan')
    if isinstance(email, str):
        if '@' in email:
            domain = email.split('@').pop()
    return domain 


### NETWORK GRAPH FUNCTIONS

def get_nodes(df:pd.DataFrame):
    
    senders = list(df['from'].dropna())
    recipients = list(df['recipients'].dropna())
    
    accts = [*senders, *recipients]
    accts = list(set(accts))
    
    nf_data = [{
            "acct": a, 
            "domain": extract_domain(a), 
            "hover": a
        } for a in accts]
    
    nf = pd.DataFrame(nf_data)
    nodes = [(row['acct'], row) for row in nf.to_dict('records') if pd.notna(row['acct'])]
    
    return nodes 


def add_edges(df:pd.DataFrame, fields:dict, graph:nx.DiGraph):
    
    email_id = fields['email_id']
    
    edges = df.groupby(['from', 'recipients'])[email_id].nunique().rename("weight").reset_index().to_dict('records')
    for e in edges:
        # e['type'] = e['relationship']
        # e['hover'] = f"{e['weight']} emails from {e['from']} {e['relationship']} {e[f]}"
        if e['from'] != e['recipients']:
            make_edge('from', 'recipients', e, graph)
                
                
def make_edge(node_field_1:str, node_field_2:str, row:dict, graph):
    node_1 = row.pop(node_field_1)
    node_2 = row.pop(node_field_2)
    graph.add_edge(node_1, node_2, **row)

    
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
        node['color'] = node_colors[community_counter % len(node_colors)]

    for edge_id in g.edges:
        edge =  g.edges[edge_id]
        source_node = g.nodes[edge_id[0]]
        target_node = g.nodes[edge_id[1]]
        pastel = edge_colors[node_colors.index(source_node['color'])]
        edge['color'] =  pastel if source_node['color'] == target_node['color'] else 'lightgray'
    return g 


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


def get_network_graph(df, fields):
    nodes = get_nodes(df)

    G = nx.DiGraph(arrow_color = 'gray', arrow_size=10)
    for n in nodes:
        G.add_node(n[0], **n[1])
    
    add_edges(df, fields, G)
    return G 

### SHINY APP ###

app_ui = ui.page_fixed(
    ui.tags.style(
        """
        .action-button { 
            margin:5px 0px 5px 0px
        }
        
        """
    ),
    ui.h2("Email Log Network Graphs"),
    ui.navset_tab_card(
        ui.nav("Email Log", 
            ui.layout_sidebar(
                ui.panel_sidebar(
                        ui.input_action_button("generate_ids", "Generate email ids"),
                        ui.input_action_button("parse_email_log", "Process log"),
                        ui.download_button("download_log", "Download log"),
                        ui.output_text_verbatim("actions") 
                ),
                ui.panel_main( 
                    ui.input_file("file1", "Upload Email Log", accept=[".csv", ".xlsx"], multiple=False, placeholder='XLSX or CSV', width="100%"),
                    ui.row(
                        ui.input_select('from_field', "From Field",choices=[], width="150px"),
                        ui.input_select('to_field', "To Field", choices=[], width="150px"),
                        ui.input_select('cc_field', "CC Field", choices=[], width= "150px"),
                        ui.input_select('email_id_field', "Email ID Field",choices=[], width="150px"),
                        ui.input_select('date_field', "Date Field", choices = [], width="150px") ,
                        ui.input_checkbox('full_emails_only', "Only use full email addresses")
                    ),
                ),
            ),

            ui.output_data_frame(id="contents"),
            value = "email_log"
        ),
        ui.nav("Network Graph", 
            ui.layout_sidebar(
                ui.panel_sidebar(

                    ui.input_action_button("build_graph", "Build graph"),                    
                    # ui.input_action_button("update_styles", "Update styles"),
                    ui.input_action_button("detect_communities", "Detect communities"),
                    ui.input_action_button("reset", "Reset"),
                    ui.download_button('export_graph', "Export interactive graph")
                ),
                ui.panel_main(
                    ui.row(
                        ui.column(5,
                            ui.row(
                                ui.input_select('node_size', "Node size", ["Total email count", "Inbound emails", "Outbound emails", "Degree centrality"], width="65%"),
                                ui.input_numeric('max_node_size', "Max size", min=3, max=100, value=30, width="30%")
                            ),
                            ui.input_select('node_color', "Node color", ["Domain", "Detect communities", "Source file(s)"]),
                            ui.input_selectize('exclude', "Exclude nodes", choices = [], multiple=True),
                        ),
                        ui.column(4, 
                            ui.br(),
                            ui.input_checkbox("filter_by_date", 'Filter by date', False),
                            ui.output_ui("date_control"),
                        ),
                        ui.column(3,
                            ui.br(),
                            ui.input_checkbox("filter_by_degree", "Filter by degree (connections)", False),
                            ui.output_ui("degree_control"),
                            ui.output_plot("degree_plot", height="100px", width="auto"),
                        
                        )
                    ),
                ),
            ),

            
            
            output_widget("sigma_graph"),
            value="graph_settings"
        ), 
        ui.nav_control(
            ui.a(
                "About This Tool",
                href="https://docs.google.com/document/d/1FXlwE534HLfS9mcTJ10Ou0KxpnG14HKs-CZbSRBIZiI/",
                target="_blank",
            )
        )
    ),
)


def server(input, output, session):
    log = reactive.Value(pd.DataFrame())
    fields = reactive.Value({
            "from": None,
            "to": None,
            "cc": None,
            "email_id": None
        })
    filename = reactive.Value()
    graph = reactive.Value(nx.DiGraph(arrow_color = 'gray', arrow_size=10))
    action_log = reactive.Value("Upload a file to start")
    update_styles_div = div("update-styles-div", )
    
    @session.download(filename="log_export.csv" )
    def download_log():
        with io.BytesIO() as buf:
            log().to_csv(buf, index=False)
            yield buf.getvalue()


    
    
    @output
    @render.text
    def actions():
        return action_log()
    
    def action(act:str):
        with reactive.isolate():
            action_log.set(action_log() + "\n" + act)
    
    
    @output
    @render.data_frame
    @reactive.event(log)
    def contents():
        if input.file1() is None:
            return pd.DataFrame()
        else: 
            return render.DataGrid(log(), filters=False)

    @output
    @render.ui 
    def date_control():
        if input.date_field() and input.filter_by_date():
            min_date = log()[input.date_field()].min()
            max_date = log()[input.date_field()].max()
            return ui.input_slider("date_slider", "", min=min_date, max=max_date, drag_range=True, value=(min_date,max_date))
    
    @output 
    @render.ui 
    def degree_control():
        max_degrees = 1000
        if input.filter_by_degree():
            max_degrees = len(nx.degree_histogram(graph())) if len(graph()) > 0 else 1000
            return ui.input_slider('degree_controls', '', min=0, max=max_degrees, value=(0, max_degrees), drag_range=True)
        
    @output
    @render.plot
    def degree_plot():
        if input.filter_by_degree():
            fig, ax = plt.subplots()
            if len(graph()) > 0:
                dh = (
                    pd.Series([n[1] for n in graph().degree])
                        .value_counts(normalize=True)
                        .reset_index()
                        .rename(columns={"index": "degree"})
                        .sort_values('degree')
                )
                ax.eventplot(positions = dh.degree, orientation="horizontal", linewidths=5)
                ax.set(
                    xlim=(0,dh['degree'].max() + 1), 
                    ylim=(1, 1)
                )
                ax.axes.get_yaxis().set_visible(False)
                ax.axes.get_xaxis().set_visible(False)
                ax.margins(0)        
            return fig
    
    @reactive.Effect
    @reactive.event(input.file1)
    def _():
        f: list[FileInfo] = input.file1()
        filename.set(f[0]['name'])
        
        action(f"File: {filename()}")
        action(f"Size: {f[0]['size']}")
        
        if f[0]['type'] == 'text/csv':
            df = pd.read_csv(f[0]['datapath'], dtype_backend='pyarrow').pipe(clean_columns)
            log.set(df)
            columns = [None, *sorted(list(df.columns))]
        elif f[0]['name'][-5:] == ".xlsx":
            df = pd.read_excel(f[0]['datapath'], dtype_backend='pyarrow').pipe(clean_columns)
            log.set(df)
            columns = [None, *sorted(list(df.columns))]
            
        action(f"{len(log())} rows")
        ui.update_select("from_field", choices = columns, selected='from' if 'from' in columns else None)
        ui.update_select("to_field",   choices = columns, selected = 'to' if 'to' in columns else None)
        ui.update_select("cc_field",   choices = columns, selected = 'cc' if 'cc' in columns else None)
        ui.update_select("email_id_field", choices = columns, selected = 'email_id' if 'email_id' in columns else None)
        ui.update_select("date_field", choices = columns, selected = 'date' if 'date' in columns else None)


    @reactive.Effect
    @reactive.event(input.generate_ids)
    def _():
        df = log().reset_index().drop(columns=['index'])
        df['email_id'] = df.apply(lambda x: str(uuid4()), axis=1)
        choices = list(df.columns)
        ui.update_select("email_id_field", choices = choices, selected = 'email_id')
        fields.set(fields().update({'email_id':'email_id'}))
        log.set(df)
        
        
    @reactive.Effect
    @reactive.event(input.parse_email_log)
    def _():
        file_fields = {
            "from": input.from_field() if input.from_field() != "" else None,
            "to": input.to_field() if input.to_field() != "" else None,
            "cc": input.cc_field() if input.cc_field() != "" else None,
            "email_id": input.email_id_field()
        }
        
        df = log().reset_index().drop(columns=['index'])
        df = parse_emails(df, file_fields)
        choices = list(df.columns)
        
        file_fields = {
            "from": "from",
            "to": "to", 
            "cc": "cc",
            "email_id": "email_id"
        }
        ui.update_select("email_id_field", choices = choices, selected = 'email_id')
        ui.update_select("from_field",choices = choices, selected='from')
        ui.update_select("to_field", choices = choices, selected='recipients')
        ui.update_select("cc_field", choices = choices, selected=None)
        
        # ui.update_select("from_field", selected='from')
        fields.set(file_fields)
        log.set(df)
        action("Cleaned.")
        
    @reactive.Effect
    @reactive.event(input.date_field)
    def _():
        if input.date_field() is not None:
            df = log()
            try:
                df['date'] = pd.to_datetime(df[input.date_field()])
                ui.update_select("date_field", choices = list(df.columns), selected = 'date')
                log.set(df)
            except Exception as e:
                print(e)
        
    @reactive.Effect
    @reactive.event(input.build_graph)
    def _():

        df = log()
        fields.set({
            "from": input.from_field() if input.from_field() != "" else None,
            "to": input.to_field() if input.to_field() != "" else None,
            "cc": input.cc_field() if input.cc_field() != "" else None,
            "email_id": input.email_id_field()
        })
        
        if input.date_field() is not None and input.filter_by_date():
            df = log().pipe(lambda df: df[df[input.date_field()].between(input.date_slider()[0], input.date_slider()[1])])
        
        print(fields())
        new_graph = get_network_graph(df, fields())
        nx.set_node_attributes(new_graph, [filename()], "files")
        
        if len(graph()) > 0 and input.filter_by_date() is False:
            old_graph = graph()
            node_data = {n: eval(old_graph.nodes[n]['files']) for n in old_graph.nodes}    
            new_graph = merge_graphs(new_graph, old_graph)
            
        node_data = {n: str(new_graph.nodes[n]['files']) for n in new_graph.nodes}
        nx.set_node_attributes(new_graph, node_data, 'files')
        graph.set(new_graph)
        
        ui.update_selectize("exclude", choices=list(graph().nodes), selected = '')


    @reactive.Effect
    def _():
        if len(graph()) > 0:
            ui.update_selectize("exclude", choices=list(graph().nodes), selected = '')

    @reactive.Effect
    @reactive.event(input.reset)
    def _():
        ui.update_select("node_size", selected = "Total email count")
        ui.update_select("node_color", selected="Domain")
        ui.update_selectize("exclude", choices=list(graph().nodes), selected = '')
        ui.update_checkbox("filter_by_date", value=False)
        ui.update_slider("filter_by_degree", value=False)
        graph.set(nx.DiGraph())
        
    @reactive.Effect
    @reactive.event(input.detect_communities)
    def _():  
        print("detecting communities")  
        new_graph = community_colors(graph())
        graph.set(new_graph)
        ui.update_selectize("node_color", selected="Detect communities")
        
                    
    @output
    @render_widget
    # @reactive.event(
    #     input.update_styles, graph
    # )
    def sigma_graph():

        filter_fn = filters.hide_nodes(input.exclude())
        view = nx.subgraph_view(graph(), filter_node=filter_fn)
        
        if input.filter_by_degree():
            included_nodes = [node for node in graph() if input.degree_controls()[0] <= nx.degree(graph(), node) <= input.degree_controls()[1]]    
            filter_fn = filters.show_nodes(included_nodes)
            view = nx.subgraph_view(view, filter_node=filter_fn )
        
        large_layout = {
            "adjustSizes": False,
            "barnesHutOptimize": True,
            "barnesHutTheta":1,
            "StrongGravityMode": False,
            "edgeWeightInfluence":.1
        } 

        small_layout = {
            "adjustSizes": False,
            "StrongGravityMode": True,
            "edgeWeightInfluence":.3
        } 
        
        node_size_map = {
            "Total email count": graph().degree,
            "Inbound emails": graph().in_degree,
            "Outbound emails": graph().out_degree,
            "Degree centrality": nx.degree_centrality(graph())
        }
        
        node_color_map = {
            "Domain": "domain", 
            "Source file(s)": "files",
            "Detect communities": "color"
        }

        
        return Sigma(
            view, 
            height=800,
            layout_settings=small_layout if len(view) < 1000 else large_layout, 
            edge_size='weight',
            edge_size_range = (0.1, 5),
            edge_weight='weight',
            edge_zindex='weight',
            edge_color = 'weight', 
            edge_color_gradient=(("#dddddd", "black")),
            max_categorical_colors = 20,
            node_size = node_size_map.get(input.node_size(), graph().degree),
            node_size_range = (3, input.max_node_size()),
            node_color= node_color_map.get(input.node_color(), "domain"),
            start_layout=30 if len(graph()) < 1000 else 120
        )


    @session.download(filename="graph_export.html")
    def export_graph():

        # if input.detect_communities() is not False:
        #     graph.set(community_colors(graph()))
    
    
        filter_fn = filters.hide_nodes(input.exclude())
        view = nx.subgraph_view(graph(), filter_node=filter_fn)
        
        if input.filter_by_degree():
            included_nodes = [node for node in graph() if input.degree_controls()[0] <= nx.degree(graph(), node) <= input.degree_controls()[1]]    
            filter_fn = filters.show_nodes(included_nodes)
            view = nx.subgraph_view(view, filter_node=filter_fn )
        
        layout = {
            "adjustSizes": False,
            "barnesHutOptimize": True,
            "barnesHutTheta":1,
            "StrongGravityMode": False,
            "edgeWeightInfluence":1
        } 
        
        node_size_map = {
            "Total email count": graph().degree,
            "Inbound emails": graph().in_degree,
            "Outbound emails": graph().out_degree,
            "Degree centrality": nx.degree_centrality(graph())
        }
        
        node_color_map = {
            "Domain": "domain", 
            "Source file(s)": "files"
        }
        with io.BytesIO() as bytes_buf:
            with io.TextIOWrapper(bytes_buf) as text_buf:
                Sigma.write_html(
                    view,
                    path=text_buf, 
                    height=800,
                    layout_settings={} if len(view) < 1000 else layout, 
                    edge_size='weight',
                    edge_size_range = (0.1, 5),
                    edge_weight='weight',
                    edge_zindex='weight',
                    edge_color = 'weight', 
                    edge_color_gradient=(("#dddddd", "black")),
                    max_categorical_colors = 20,
                    node_size = node_size_map.get(input.node_size(), graph().degree),
                    node_size_range = (3, input.max_node_size()),
                    node_color= node_color_map.get(input.node_color(), "domain"),
                    start_layout=30 if len(graph()) < 1000 else 120
                )
                yield bytes_buf.getvalue()
    
app = App(app_ui, server)

