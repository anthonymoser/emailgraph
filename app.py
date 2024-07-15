from shiny import App, reactive, render, ui
from shiny.ui import fill
from shinywidgets import output_widget, render_widget
from shiny.types import FileInfo
from htmltools import TagList, div

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
                ui.input_action_button("filter_log_to_graph_selection", "Filter log by graph selection"),
                ui.input_action_button("filter_graph_to_log_selection", "Filter graph by log selection"),
                
                ui.input_checkbox("filter_by_date", 'Filter by date', False),
                ui.output_ui("date_control"),
                ui.input_checkbox("filter_by_degree", "Filter by degree (connections)", False),
                ui.output_ui("degree_control"),
                ui.output_plot("degree_plot", height="100px", width="auto"),
            ),
            ui.accordion_panel(
                "STYLE",
                ui.input_select('node_size', "Node size", ["Total email count", "Inbound emails", "Outbound emails", "Degree centrality"]),
                ui.input_select('node_color', "Node color", ["Domain", "Detect communities", "Source file(s)"]),
                ui.input_select('edge_color', "Edge color", ["Grayscale", "Email count", "Recipient Type", "Sender Domain"], selected = "Grayscale"),
                ui.input_action_button("detect_communities", "Detect communities"),
            ),
            id = "controls_accordion"
        ),
        ui.input_action_button("reset", "Reset Graph"),
        open="always"
    ),


    ui.navset_underline(
        ui.nav_menu("Save / Load",
            ui.nav_control(
                ui.download_button("download_log", "Download log"),
                ui.download_button('export_graph', "Export graph HTML"),
                ui.download_button("save_graph_data", "Save QNG Graph File"),
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
            ui.output_data_frame(id="contents"),
        ),
        multiple=True,
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
        
        .accordion .action-button {
            --bs-btn-margin-y: 0px;
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
    G = reactive.Value(nx.MultiDiGraph(arrow_color = 'gray', arrow_size=10))
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
    viz = reactive.value()
    
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
            "Recipient Type": "type",
            "Sender Domain": "sender_domain"
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
        else: 
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
            log()['uuid'] = log().index.map(lambda _: uuid4())
            fields['id'] = 'uuid'
        field_map.set(fields)
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
        ui.update_accordion_panel(id="controls_accordion", target="FILTER", show=True)
        
        
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
        print(len(graph_df()))
        graph = build_graph(G(), graph_factory, graph_df())
        G.set(graph)


    @reactive.effect
    @reactive.event(input.edge_color)
    def _():
        if input.edge_color() == "Grayscale":
            SF.set(
                SigmaFactory(
                    clickable_edges=True, 
                    edge_size='emails',
                    edge_color = 'emails',  
                    edge_color_gradient=("#d3d3d3", "#969696")
                )
            )
        elif input.edge_color() == "Email count":
            SF.set(
                SigmaFactory(
                    clickable_edges=True, 
                    edge_size='emails',
                    edge_color = 'emails'
                )
            )
        elif input.edge_color() == "Recipient Type":
            SF.set(
                SigmaFactory(
                    clickable_edges=True, 
                    edge_size='emails',
                    edge_color = 'type'
                )
            )
        elif input.edge_color() == "Sender Domain": 
            SF.set(
                SigmaFactory(
                    clickable_edges = True,
                    edge_size = 'emails',
                    edge_color = "sender_domain",
                    edge_color_palette = node_colors()
                )
            )
        
    # Make graph widget
    @reactive.effect
    @reactive.event(G, SF) 
    def graph_widget():
        try:
            nc = get_node_colors(G())
            node_colors.set(nc)
            # edge_colors = get_edge_colors()
            layout = viz().get_layout()
            camera_state = viz().get_camera_state()
            # viz.set(SF().make_sigwma(G(), node_colors, edge_colors, layout = layout, camera_state = camera_state))
            viz.set(
                SF().make_sigma(
                    G(), 
                    node_colors = nc,
                    layout = layout, 
                    camera_state = camera_state,
                )
            )
        except Exception as e:
            print(e)
            # viz.set(SF().make_sigma(G(), node_colors, edge_colors))
            viz.set(
                SF().make_sigma(
                    G(), 
                )
            )

    # Update visualization
    @render_widget()
    def sigma_graph():
        return viz()
    
    @reactive.Effect
    def _():
        if len(G()) > 0:
            ui.update_selectize("exclude", choices=list(G().nodes), selected = '')

    @reactive.Effect
    @reactive.event(input.reset)
    def _():
        ui.update_select("node_size", selected = "Total email count")
        ui.update_select("node_color", selected="Domain")
        ui.update_selectize("exclude", choices=list(G().nodes), selected = '')
        ui.update_checkbox("filter_by_date", value=False)
        ui.update_slider("filter_by_degree", value=False)
        log.set(backup_log())
        
        
    @reactive.Effect
    @reactive.event(input.detect_communities)
    def _():  
        print("detecting communities")  
        new_graph = community_colors(G())
        graph.set(new_graph)
        ui.update_selectize("node_color", selected="Detect communities")
        
                    
    # @output
    # @render_widget
    # def sigma_G():

    #     filter_fn = filters.hide_nodes(input.exclude())
    #     view = nx.subgraph_view(G(), filter_node=filter_fn)
        
    #     if input.filter_by_degree():
    #         included_nodes = [node for node in G() if input.degree_controls()[0] <= nx.degree(G(), node) <= input.degree_controls()[1]]    
    #         filter_fn = filters.show_nodes(included_nodes)
    #         view = nx.subgraph_view(view, filter_node=filter_fn )
        
    #     large_layout = {
    #         "adjustSizes": False,
    #         "barnesHutOptimize": True,
    #         "barnesHutTheta":1,
    #         "StrongGravityMode": False,
    #         "edgeWeightInfluence":.1
    #     } 

    #     small_layout = {
    #         "adjustSizes": False,
    #         "StrongGravityMode": True,
    #         "edgeWeightInfluence":.3
    #     } 
        
    #     node_size_map = {
    #         "Total email count": G().degree,
    #         "Inbound emails": G().in_degree,
    #         "Outbound emails": G().out_degree,
    #         "Degree centrality": nx.degree_centrality(G())
    #     }
        
    #     node_color_map = {
    #         "Domain": "domain", 
    #         "Source file(s)": "files",
    #         "Detect communities": "color"
    #     }

        
    #     return Sigma(
    #         view, 
    #         height=1000,
    #         layout_settings=small_layout if len(view) < 1000 else large_layout, 
    #         edge_size='weight',
    #         edge_size_range = (0.1, 5),
    #         edge_weight='weight',
    #         edge_zindex='weight',
    #         edge_color = 'weight', 
    #         edge_color_gradient=(("#dddddd", "black")),
    #         max_categorical_colors = 20,
    #         node_size = node_size_map.get(input.node_size(), G().degree),
    #         node_size_range = (3, input.max_node_size()),
    #         node_color= node_color_map.get(input.node_color(), "domain"),
    #         start_layout=30 if len(G()) < 1000 else 120
    #     )


    # @reactive.Effect
    # @reactive.event(input.date_field)
    # def _():
    #     if input.date_field() is not None:
    #         df = log()
    #         try:
    #             df['date'] = pd.to_datetime(df[input.date_field()])
    #             ui.update_select("date_field", choices = list(df.columns), selected = 'date')
    #             log.set(df)
    #         except Exception as e:
    #             print(e)
        
    # @reactive.Effect
    # @reactive.event(input.build_graph)
    # def _():

    #     df = log()
    #     fields.set({
    #         "from": input.from_field() if input.from_field() != "" else None,
    #         "to": input.to_field() if input.to_field() != "" else None,
    #         "cc": input.cc_field() if input.cc_field() != "" else None,
    #         "email_id": input.email_id_field()
    #     })
        
    #     if input.date_field() is not None and input.filter_by_date():
    #         df = log().pipe(lambda df: df[df[input.date_field()].between(input.date_slider()[0], input.date_slider()[1])])
        
    #     print(fields())
    #     new_graph = get_network_graph(df, fields())
    #     nx.set_node_attributes(new_graph, [filename()], "files")
        
    #     if len(G()) > 0 and input.filter_by_date() is False:
    #         old_graph = G()
    #         node_data = {n: eval(old_graph.nodes[n]['files']) for n in old_graph.nodes}    
    #         new_graph = merge_graphs(new_graph, old_graph)
            
    #     node_data = {n: str(new_graph.nodes[n]['files']) for n in new_graph.nodes}
    #     nx.set_node_attributes(new_graph, node_data, 'files')
    #     graph.set(new_graph)
        
    #     ui.update_selectize("exclude", choices=list(G().nodes), selected = '')
    @render.download(filename=f"email_graph.qng")
    def save_graph_data():
        adj = nx.to_dict_of_dicts(G())
        attrs = { n: G().nodes[n] for n in G().nodes()}
        qng = QNG(adjacency=adj, node_attrs=attrs, sigma_factory=SF())
        yield msgspec.json.encode(qng)

    @render.download(filename="log_export.csv" )
    def download_log():
        with io.BytesIO() as buf:
            log().to_csv(buf, index=False)
            yield buf.getvalue()
            
    @render.download(filename="graph_export.html")
    def export_graph():

        # if input.detect_communities() is not False:
        #     graph.set(community_colors(G()))
    
    
        filter_fn = filters.hide_nodes(input.exclude())
        view = nx.subgraph_view(G(), filter_node=filter_fn)
        
        if input.filter_by_degree():
            included_nodes = [node for node in G() if input.degree_controls()[0] <= nx.degree(G(), node) <= input.degree_controls()[1]]    
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
            "Total email count": G().degree,
            "Inbound emails": G().in_degree,
            "Outbound emails": G().out_degree,
            "Degree centrality": nx.degree_centrality(G())
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
                    node_size = node_size_map.get(input.node_size(), G().degree),
                    node_size_range = (3, input.max_node_size()),
                    node_color= node_color_map.get(input.node_color(), "domain"),
                    start_layout=30 if len(G()) < 1000 else 120
                )
                yield bytes_buf.getvalue()
    
app = App(app_ui, server)

