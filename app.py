from shiny import App, reactive, render, ui
from shiny.ui import fill
from shinywidgets import output_widget, render_widget
from shiny.types import FileInfo
from htmltools import TagList, div

from copy import copy 
import msgspec
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from networkx.classes import filters
import io
from uuid import uuid4
from ipysigma import Sigma
from util import * 
from qng import NodeFactory, LinkFactory, GraphFactory, SigmaFactory, Element, QNG, load_schema

graph_schema = load_schema('email_log_schema.qngs')
graph_factory = GraphFactory(
    node_factories= list(dict(graph_schema.node_factories).values()), 
    link_factories= graph_schema.link_factories
)

### SHINY APP ###

app_ui = ui.page_sidebar(
    ui.sidebar("",
        ui.accordion(
            ui.accordion_panel(
                "UPLOAD",
                ui.input_file("file1", "Upload Email Log", accept=[".csv", ".xlsx"], multiple=False, placeholder='XLSX or CSV'),
                ui.input_select('from_field', "Sender",choices=[]),
                ui.input_selectize('to_field', "Recipients", choices=[], multiple=True),
                ui.input_select('email_id_field', "Email ID (optional)",choices=[]),
                ui.input_select('date_field', "Date Field (optional)", choices = []),
                ui.input_checkbox('full_emails_only', "Only use full email addresses"),
                ui.input_action_button("add_to_graph", "Add to graph")
            ),
            ui.accordion_panel(
                "SELECT",
                ui.input_selectize("selected_nodes", "", choices=[], multiple=True, width="100%"),  
                    ui.row(
                        ui.layout_columns(
                            ui.input_action_button("combine", "Merge"),
                            ui.input_action_button("remove", "Remove"),
                            col_widths = (6,6)                                    
                        ),
                    ), 
            ),
            ui.accordion_panel(
                "FILTER",
                ui.input_action_button("filter_graph_to_log_selection", "Filter graph (match log)"),
                ui.input_action_button("filter_log_to_graph_selection", "Filter log (match graph)"),
                ui.input_date_range("date_filter", "Filter by date field"),
                # ui.output_ui("degree_control"),
                # ui.output_plot("degree_plot", height="100px", width="auto"),
            ),
            ui.accordion_panel(
                "STYLE",
                ui.input_select('node_size', "Node size", ["Total email count", "Inbound emails", "Outbound emails", "Degree centrality"], selected = "Total email count"),
                ui.input_select('node_color', "Node color", ["Domain", "Detect communities", "Source file(s)"]),
                ui.input_select('edge_color', "Edge color", ["Grayscale", "Email count", "Recipient type", "Sender domain", "Inferred community"], selected = "Grayscale"),
                # ui.input_action_button("detect_communities", "Detect communities"),
            ),
            id = "controls_accordion"
        ),
        ui.input_action_button("reset", "Reset"),
        open="always"
    ),


    ui.navset_underline(
        ui.nav_menu("Save / Load",
            ui.nav_control(
                ui.download_button("download_log", "Download filtered log (CSV)"),
                ui.download_button("save_graph_data", "Download graph (QNG)"),
                ui.download_button('export_graph', "Export graph (HTML)"),
                ui.input_file("graph_upload", "", accept=[".qng"], multiple=True, placeholder='.QNG', width="100%", button_label = 'Upload graph (QNG)' ),
            ),
            ui.nav_control(
                ui.a(
                    "About This Tool",
                    href="https://docs.google.com/document/d/1FXlwE534HLfS9mcTJ10Ou0KxpnG14HKs-CZbSRBIZiI/",
                    target="_blank",
                )
            )    
        )
    ),

    ui.accordion(
        ui.accordion_panel(
            "LOG VIEW",
            ui.output_data_frame(id="contents")
        ),
        multiple=True,
        open = None,
        id = "output_accordion", 
    ),
    fill.as_fillable_container(
        ui.div(
            fill.as_fill_item(output_widget("sigma_graph")),
            {"style": "flex: 1;"},
            id = "sigma_graph_div",
        )
    ),
    ui.tags.style("""
        :root { 
            --bslib-mb-spacer: .1rem; 
            --bs-border-radius: 0;
              font-size: small;
        },
        .action-button { 
            margin:5px 0px 5px 0px;
            border-radius: 0;
        },
        
        .accordion .btn-default .action-button {
            --bs-btn-margin-y: -1px;
            --bs-btn-border-color: #acacac;
            margin: 0px 0px -1px 0px;
        }
        
        .nav-item .btn {
            --bs-btn-border-width: 0;
            --bs-btn-padding-x: 1em;
            --bs-btn-hover-bg: var(--bs-tertiary-bg);
            --bs-btn-hover-color: #000000;
            --bs-btn-padding-y: 0;
            --bs-btn-font-weight: 0;
        } 
        
        .nav .input-group .form-control {
            display: none;
        }
        
        .bslib-sidebar-layout > .sidebar .shiny-input-container {
            width: 100%;
            margin-top: .75rem;
        }
        .accordion-button:not(.collapsed) {
            background-color: #f2f2f2;
        }
    """),
    {"style": "display:flex; flex-direction: column;"},
    title = "Email Log Network Graphs",
    height="auto",
)


def server(input, output, session):
    
    
    ### INITIALIZE 
    # Log is the uploaded file
    log = reactive.Value(pd.DataFrame())
    
    # Used to store the log when graph filters are applied
    backup_log = reactive.Value(pd.DataFrame())
    
    # Parsed log is the file after extracting email addresses and separating each recipient into rows (still filterable by id)
    parsed_log = reactive.Value(pd.DataFrame())
    
    # graph_df is the parsed log rolled up into counts between nodes
    graph_df = reactive.Value(pd.DataFrame())
    
    filename = reactive.Value()
    field_map = reactive.Value({})
    G = reactive.Value(nx.MultiDiGraph(arrow_color = 'gray', arrow_size=5))
    filtered_data = reactive.Value(pd.DataFrame())
    node_colors = reactive.Value()
    
    ### Factories
    SF = reactive.value(
        SigmaFactory(
            clickable_edges=True, 
            edge_size='emails',
            edge_color = 'emails',  
            edge_color_gradient=("#d3d3d3", "#969696")
        )
    )
    viz = reactive.Value()
    
    # maybe deprecated?
    update_styles_div = div("update-styles-div")
    

    # Update node list 
    @reactive.effect
    def _():
        node_names = get_node_names(G())
        choices = {node_names[n]: n for n in sorted(list(node_names.keys()))}
        ui.update_selectize("selected_nodes", choices= choices) 
        
    #### GRAPH CONTROLS ###
    @reactive.Effect
    @reactive.event(input.filter_graph_to_log_selection)
    def _():
        ds = data_selected = contents.data_view(selected=True)
        filtered_ids = ds[field_map()['id']].unique()
        filtered_rows = parsed_log().pipe( lambda df: df[ df['id'].isin(filtered_ids) ])
        rolled_up = count_emails(filtered_rows)
        G.set(nx.MultiDiGraph(arrow_color = 'gray', arrow_size=10))
        graph_df.set(rolled_up)
        
    @reactive.Effect
    @reactive.event(input.filter_log_to_graph_selection)
    def _():
        selected_ids = get_email_ids_from_graph_selection()
        filtered_log = log().pipe(lambda df: df[df[field_map()['id']].isin(selected_ids)])
        log.set(filtered_log)
    
    # Remove selected nodes
    @reactive.effect
    @reactive.event(input.remove)
    def _():
        graph = G().copy()
        graph.remove_nodes_from(get_selected_nodes())
        G.set(graph)
    
    ### Combine selected nodes
    @reactive.effect
    @reactive.event(input.combine)
    def _():
        selected = get_selected_nodes()
        print("Combining nodes: ", selected)
        new_graph = combine_nodes(G(), selected)
        # merged.set( merged() + [selected])    
        G.set(new_graph)
        
    def get_selected_nodes():
        try:
            # Selected by dropdown
            selected = [ n for n in input.selected_nodes() ]
            
            # Selected by graph
            if viz().get_selected_node() is not None:
                selected += [ viz().get_selected_node() ]
                
            print("SELECTED NODES: ", selected)
            return selected  
        except Exception as e:
            return []
            
    def get_email_ids_from_graph_selection():
        selected_ids = []
        pl = parsed_log()
        
        # Selected node
        selected_node = viz().get_selected_node()
        if selected_node is not None:
            aliases = G.nodes[selected_node].get('alias_ids', [ selected_node ])
            selected_node_filter = (
                ( pl['to'].isin(selected_node) ) |
                ( pl['from'].isin(selected_node) )
            )
            selected_ids = [ *selected_ids, *list(pl[selected_node_filter].id.unique()) ]
        
        # Selected edge
        selected_edge = viz().get_selected_edge()
        if selected_edge is not None:
            selected_edge_filter = (
                ( (pl['to'] == selected_edge[0]) & (pl['from'] == selected_edge[1])) |
                ( (pl['from'] == selected_edge[0]) & (pl['to'] == selected_edge[1]))
            )
            selected_ids = [ *selected_ids, *list(pl[selected_edge_filter].id.unique()) ]
        
        # Selected edge types
        edge_color_fields = {
            "Grayscale": "emails",
            "Email count": "emails", 
            "Recipient type": "type",
            "Sender domain": "sender_domain"
        }
        edge_filter = viz().get_selected_edge_category_values()
        if edge_filter is not None:
            edge_filter_conditions = (pl[edge_color_fields.get(input.edge_color(), 'type')].isin(edge_filter))
            selected_ids = [ *selected_ids, *list(pl[edge_filter_conditions].id.unique()) ]
        
        # Selected node types
        node_filter = viz().get_selected_node_category_values()
        if node_filter is not None:
            node_filter_conditions = (
                (pl.sender_domain.isin(node_filter)) | 
                (pl.recipient_domain.isin(node_filter)) 
            )
            selected_ids = [ *selected_ids, *list(pl[node_filter_conditions].id.unique()) ]
        
        selected_ids = list(set(selected_ids))
        return selected_ids 
            
    @output 
    @render.ui 
    def degree_control():
        max_degrees = 1000
        if input.filter_by_degree():
            max_degrees = len(nx.degree_histogram(G())) if len(G()) > 0 else 1000
            return ui.input_slider('degree_controls', '', min=0, max=max_degrees, value=(0, max_degrees), drag_range=True)
        
    @output
    @render.plot
    def degree_plot():
        if input.filter_by_degree():
            fig, ax = plt.subplots()
            if len(G()) > 0:
                dh = (
                    pd.Series([n[1] for n in G().degree])
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
    
    # On file upload
    @reactive.Effect
    @reactive.event(input.graph_upload)
    def _():
        f: list[FileInfo] = input.graph_upload()
        datapath = f[0]['datapath']
        
        if f[0]['type'] == "application/octet-stream":
            with open(datapath, 'r') as f:
                graph_data = msgspec.json.decode(f.read(), type=QNG)
                mg = graph_data.multigraph()
                
                if len(G()) > 0:
                    G.set(nx.compose(G(), mg))
                else:
                    G.set(mg)
    
    @reactive.Effect
    @reactive.event(input.file1)
    def _():
        f: list[FileInfo] = input.file1()
        filename.set(f[0]['name'])
        
        if f[0]['type'] == 'text/csv':
            df = pd.read_csv(f[0]['datapath'], dtype_backend='pyarrow').pipe(clean_columns)
            log.set(df)
            backup_log.set(df)
            columns = [None, *sorted(list(df.columns))]
        elif f[0]['name'][-5:] == ".xlsx":
            df = pd.read_excel(f[0]['datapath'], dtype_backend='pyarrow').pipe(clean_columns)
            log.set(df)
            backup_log.set(df)
            columns = [None, *sorted(list(df.columns))]
        
        ui.update_select("from_field", choices = columns, selected = match_column(columns, "from") )
        ui.update_select("to_field",   choices = columns, selected = match_columns(columns, ["to", "cc", "bcc"]))
        ui.update_select("email_id_field", choices = columns, selected = match_column(columns, "_id"))
        ui.update_select("date_field", choices = columns, selected = match_column(columns, "date"))

    @output
    @render.data_frame
    @reactive.event(log)
    def contents():
        if input.file1() is None:
            return pd.DataFrame()
        elif len(log()) > 0: 
            ui.update_accordion_panel(id="primary_accordion", target="LOG VIEW", show=True)
            return render.DataGrid(log(), filters=True)
    
    @reactive.effect
    @reactive.event(input.add_to_graph)
    def add_to_graph_clicked():
        fields = {
            "from": input.from_field(),
            "to": input.to_field(),
            "id": input.email_id_field(),
            "date": input.date_field()
        }
        print(fields)
        if fields['id'] == "":
            log()['uuid'] = log().index.map(lambda _: str(uuid4()))
            fields['id'] = 'uuid'
        field_map.set(fields)
        
        if fields['date'] != "":
            df = log().copy()
            df[fields['date']] = pd.to_datetime(df[fields['date']])
            df['date_field'] = pd.to_datetime(df[fields['date']]).dt.strftime("%Y-%m-%d")
            min_date = df[fields['date']].min()
            max_date = df[fields['date']].max()
            print(min_date, max_date)
            
            ui.update_date_range(
                "date_filter", 
                start = min_date, 
                end = max_date,
                min = min_date,
                max = max_date
            )
            
            log.set(df)
        
        # Begin async reprocessing of the log
        add_log(log(), filename(), fields, input.full_emails_only())
    
    
    @ui.bind_task_button(button_id = 'add_to_graph')
    @reactive.extended_task
    async def add_log(df:pd.DataFrame, filename:str, fields:dict, full_emails_only:bool) ->pd.DataFrame:
        print("Adding log")
        try:
            rfl = reformat_log(df, fields, filename, full_emails_only)
            return rfl 
        except Exception as e:
            print(e)
    
    
    @reactive.effect
    @reactive.event(add_log.result)
    def _():
        rfl = pd.concat([parsed_log(), add_log.result()])
        print(f"Processed into {len(rfl)} rows")
        parsed_log.set(rfl)
        ui.update_accordion_panel(id="controls_accordion", target="UPLOAD", show=False)
        ui.update_accordion_panel(id="controls_accordion", target="SELECT", show=True)
        ui.update_accordion_panel(id="controls_accordion", target="FILTER", show=True)
        
        
    @reactive.effect
    @reactive.event(input.date_filter)
    def _():
        if len(field_map()) > 0:
            start = input.date_filter()[0]
            end = input.date_filter()[1]
            if start is not None and end is not None:
                df = log().copy()
                date_field = field_map()['date']
                df = df[ 
                    (df[date_field] >= pd.to_datetime(start)) &
                    (df[date_field] <= pd.to_datetime(end))
                ]
                log.set(df) 
        
        
    @reactive.effect
    @reactive.event(parsed_log, input.reset)
    def _():
        if len(parsed_log()) > 0:
            rolled_up = count_emails(parsed_log())
            graph_df.set(rolled_up)
        print(len(graph_df()))
    
    
    @reactive.effect
    @reactive.event(graph_df)
    def _():
        graph = build_graph(G(), graph_factory, graph_df())
        G.set(graph)


    ### Make graph widget
    @reactive.effect
    @reactive.event(G, SF) 
    def graph_widget():
        try:
            layout = viz().get_layout()
            camera_state = viz().get_camera_state()
            viz.set(
                SF().make_sigma(
                    G(), 
                    layout = layout, 
                    camera_state = camera_state
                )
            )
        except Exception as e:
            print(e)
            viz.set(
                SF().make_sigma(
                    G(),
                )
            )


    @reactive.effect
    @reactive.event(input.node_size, G)
    def _():
        node_size_map = {
            "Total email count": dict(G().degree),
            "Inbound emails": dict(G().in_degree),
            "Outbound emails": dict(G().out_degree),
            "Degree centrality": nx.degree_centrality(G())
        }
        new_sf = copy(SF())
        new_sf.node_size = node_size_map.get(input.node_size())
        SF.set(new_sf)
    
    
    @reactive.effect
    @reactive.event(input.node_color, G)
    def _():
        node_color_map = {
            "Domain": "type", 
            "Source file(s)": "filename",
            "Detect communities": "community"
        }
        new_sf = copy(SF())
        selected = node_color_map.get(input.node_color())
        new_sf.node_color = selected
        
        nc = get_node_colors(G(), selected)
        node_colors.set(nc)
        new_sf.node_color_palette = nc        

        match input.node_color():
            case "Detect communities":    
                new_graph = community_colors(G())
                new_sf.node_color = None 
                new_sf.raw_node_color = "community"
                viz.set(new_sf.make_sigma(new_graph))
                # new_sf.node_color_palette = None
                # G.set(new_graph)
                # ui.update_select("edge_color", selected = "Inferred community")
            
        SF.set(new_sf)
        
        
    @reactive.effect
    @reactive.event(input.edge_color, G)
    def _():
        edge_color_map = {
            "Grayscale": "emails", 
            "Email count": "emails",
            "Recipient type": "type",
            "Sender domain": "sender_domain", 
            "Inferred community": "community"
        }
        new_sf = copy(SF())
        new_sf.edge_color = edge_color_map.get(input.edge_color())
        if input.edge_color() == "Grayscale":
            new_sf.edge_color_gradient = ("#d3d3d3", "#969696")
            new_sf.edge_color_palette = None
        else:
            new_sf.edge_color_gradient = None
            # new_sf.edge_color_palette = node_colors()
        SF.set(new_sf)


    # Update visualization
    @render_widget()
    def sigma_graph():
        return viz()

    @reactive.Effect
    @reactive.event(input.reset)
    def _():
        ui.update_select("node_size", selected = "Total email count")
        ui.update_select("node_color", selected="Domain")
        ui.update_select("edge_color", selected = "Grayscale")
        log.set(backup_log())

     
    @render.download(filename=f"email_graph.qng")
    def save_graph_data():
        adj = nx.to_dict_of_dicts(G())
        attrs = { n: G().nodes[n] for n in G().nodes()}
        qng = QNG(adjacency=adj, node_attrs=attrs, sigma_factory=SF())
        yield msgspec.json.encode(qng)

    @render.download(filename="log_export.csv" )
    def download_log():
        with io.BytesIO() as buf:
            ds = data_selected = contents.data_view(selected=True)
            ds.to_csv(buf, index=False)
            yield buf.getvalue()
            
    @render.download(filename="graph_export.html")
    def export_graph():
        return SF().export_graph(G())
    
app = App(app_ui, server)

