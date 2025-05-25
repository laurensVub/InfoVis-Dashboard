#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# =============================
# Dataset Loading
# =============================


final_df = pd.read_csv("Data/Dataset.csv")


# In[2]:


final_df.head()
final_df.columns
print(final_df.shape)
print(final_df[['id', 'name']].head())



# In[3]:


import dash
from dash import dcc, html, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
import threading
import webbrowser
import pandas as pd              # Used by callbacks
import plotly.graph_objects as go
import squarify
import statsmodels.api as sm     # Used in trends regression
from dash.exceptions import PreventUpdate

# -------------------------
# Dash app
# -------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Dashboard"

# -------------------------
# Helper builders
# -------------------------
def build_top_control(active_tab: str):
    return dbc.ButtonGroup(
        [
            dbc.Button(
                "Games", id="games-button", n_clicks=0,
                color="primary" if active_tab == "games" else "secondary",
                size="sm",  # Smaller button size
                style={"padding": "0.3rem 0.6rem", "fontSize": "0.9rem"}  # Reduced padding and font size
            ),
            dbc.Button(
                "Trends", id="trends-button", n_clicks=0,
                color="primary" if active_tab == "trends" else "secondary",
                size="sm",
                style={"padding": "0.3rem 0.6rem", "fontSize": "0.9rem"}
            )
        ],
        id="top_control",
        style={"width": "100%", "marginBottom": "0.5rem"}  # Reduced margin
    )

def build_search_bar(theme: str = "light"):
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Input(
                    id="search_bar",
                    placeholder="Search...",
                    type="text",
                    debounce=False
                ),
                html.Div(
                    id="search_results",
                    style={
                        "position": "absolute", "zIndex": 1000,
                        "width": "100%",
                        "backgroundColor": "#fff" if theme == "light" else "#212529",
                        "border": "1px solid #ced4da" if theme == "light" else "1px solid #495057",
                        "borderRadius": "0.25rem",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.2)",
                        "maxHeight": "200px", "overflowY": "auto",
                        "marginTop": "2px",
                        "color": "#000" if theme == "light" else "#fff"
                    }
                )
            ]
        ),
        style={
            "position": "relative",
            "marginTop": "1rem", "marginBottom": "1rem",
            "backgroundColor": "#fff" if theme == "light" else "#212529"
        }
    )

def generate_selected_game_icons(game_ids, df, colors):
    thumbs = []
    selected_df = df[df["id"].isin(game_ids)]
    for idx, gid in enumerate(game_ids):
        row = selected_df[selected_df["id"] == gid].iloc[0]
        thumbs.append(
            html.Div(
                [
                    html.Img(
                        src=row.get("background_image", ""),
                        style={
                            "width": "32px", "height": "32px", "objectFit": "cover",
                            "borderRadius": "50%",
                            "border": f"2px solid {colors[idx % len(colors)]}",
                            "boxShadow": "0 0 2px rgba(0,0,0,0.2)",
                            "cursor": "default"
                        },
                        title=row.get("name", "Unnamed Game")
                    )
                ],
                style={"position": "relative"}
            )
        )
    return thumbs

def build_compare_search_bar(theme: str = "light"):
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Input(
                    id="compare_search_bar",
                    placeholder="Search games to add…",
                    type="text", debounce=False,
                    style={"width": "100%"}
                ),
                html.Div(
                    id="compare_search_results",
                    style={
                        "position": "absolute", "zIndex": 1000,
                        "width": "100%",
                        "backgroundColor": "#fff" if theme == "light" else "#212529",
                        "border": "1px solid #ced4da" if theme == "light" else "1px solid #495057",
                        "borderRadius": "0.25rem",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.2)",
                        "maxHeight": "200px", "overflowY": "auto",
                        "marginTop": "2px",
                        "color": "#000" if theme == "light" else "#fff"
                    }
                )
            ],
            style={"padding": "0.4rem"}
        ),
        style={
            "width": "220px", "maxWidth": "220px",
            "backgroundColor": "#fff" if theme == "light" else "#212529",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.2)",
            "border": "1px solid #ced4da" if theme == "light" else "1px solid #495057",
            "borderRadius": "0.5rem"
        }
    )

def build_games_middle(selected_sort_options: list):
    sort_options = ["Added to library", "Player Rating", "YouTube Views", "Twitch Views", "Metacritic score"]
    sort_buttons = [
        dbc.Checkbox(
            label=s,
            id={"type": "sort-button", "index": s},
            value=s in selected_sort_options,
            style={"marginBottom": "0.4rem"}
        )
        for s in sort_options
    ]
    return html.Div([
        html.H6("Sort Heatmap By"),
        *sort_buttons,
        html.Div(
            [
                html.Label("Select Genre(s)", style={"fontWeight": "bold", "marginTop": "1rem"}),
                dcc.Dropdown(
                    id={"type": "genre-dropdown", "context": "games"},
                    options=[], multi=True,
                    placeholder="Filter by genre..."
                )
            ],
            style={"marginTop": "1rem"}
        )
    ])

def build_loading_component(text="Loading...", theme: str = "light"):
    return html.Div(
        [
            dbc.Spinner(
                children=[],
                size="lg",
                color="primary",
                type="border",
                spinner_style={"width": "3rem", "height": "3rem"}
            ),
            html.H4(text, className="mt-3", style={"color": "#666" if theme == "light" else "#ccc"})
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "height": "100%",
            "backgroundColor": "#f8f9fa" if theme == "light" else "#212529"
        }
    )

def build_data_view(theme: str = "light"):
    return html.Div(
        [
            html.Div(
                id="main_graph",
                children=[
                    html.Div(id="games_graph", children=build_loading_component("Loading games...", theme), style={"display": "none"}),
                    html.Div(id="trends_graph", children=build_loading_component("Loading trends...", theme), style={"display": "none"})
                ],
                style={
                    "minHeight": 0,
                    "position": "relative",
                    "width": "100%",
                    "height": "100%",  # Ensure it takes the full height of its parent
                    "overflow": "hidden",
                    "transition": "flex 0.3s ease"
                }
            ),
            html.Div(
                id="comparison_panel",
                style={
                    "display": "none",
                    "overflowY": "auto",
                    "borderTop": "1px solid #dee2e6" if theme == "light" else "1px solid #495057",
                    "backgroundColor": "#f8f9fa" if theme == "light" else "#212529",
                    "transition": "flex 0.3s ease"
                }
            ),
            html.Div(
                id="game-details-panel",
                style={
                    "position": "absolute",
                    "top": 0,
                    "right": 0,
                    "width": "360px",
                    "height": "100vh",
                    "backgroundColor": "#f8f9fa" if theme == "light" else "#212529",
                    "borderLeft": "1px solid #dee2e6" if theme == "light" else "1px solid #495057",
                    "overflowY": "auto",
                    "boxShadow": "-2px 0 8px rgba(0,0,0,0.15)",
                    "transition": "transform 0.3s ease-in-out",
                    "transform": "translateX(100%)",
                    "zIndex": 1000,
                    "padding": "1rem"
                }
            )
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": "100%",  # Ensure the container takes the full height
            "position": "relative"
        }
    )

def build_sidebar(active_tab: str, selected_sort_options: list, theme: str = "light"):
    games_controls = html.Div(
        build_games_middle(selected_sort_options),
        style={"display": "block" if active_tab == "games" else "none"}
    )

    games_year_slider = html.Div(
        [
            html.Label("Select Release Year Range", style={"fontWeight": "bold", "marginTop": "1rem"}),
            dcc.RangeSlider(
                id="games-year-range-slider",
                min=1970, max=2024, step=1,
                marks={y: str(y) for y in range(1970, 2025, 10)},
                value=[2000, 2020],
                tooltip={"placement": "bottom", "always_visible": False},
                allowCross=False,
                updatemode="mouseup"
            )
        ],
        style={
            "marginTop": "1.5rem", "padding": "1rem",
            "backgroundColor": "#f1f3f5" if theme == "light" else "#343a40",
            "borderRadius": "0.5rem",
            "boxShadow": "inset 0 1px 3px rgba(0,0,0,0.1)", "width": "100%",
            "display": "block" if active_tab == "games" else "none"
        }
    )

    games_num_slider = html.Div(
        [
            html.Label("Top number of games (3 – 200)", style={"fontWeight": "bold", "marginTop": "1rem"}),
            dcc.RangeSlider(
                id="games-num-games-slider",
                min=1, max=200, step=1, value=[1, 50],
                marks={i: str(i) for i in range(10, 201, 30)},
                tooltip={"placement": "bottom", "always_visible": False},
                allowCross=False,
                updatemode="mouseup"
            )
        ],
        style={
            "marginTop": "1.5rem", "padding": "1rem",
            "backgroundColor": "#f1f3f5" if theme == "light" else "#343a40",
            "borderRadius": "0.5rem",
            "boxShadow": "inset 0 1px 3px rgba(0,0,0,0.1)", "width": "100%",
            "display": "block" if active_tab == "games" else "none"
        }
    )

    trends_controls = html.Div(
        [
            html.Label("Show games by", style={"fontWeight": "bold", "fontSize": "1.1rem", "marginBottom": "0.5rem"}),
            dbc.Checkbox(id={"type": "trends-metric", "index": "rating"},   label="Player Rating",      value=True,  style={"marginLeft":"0.25rem","marginBottom":"0.6rem"}),
            dbc.Checkbox(id={"type": "trends-metric", "index": "added"},    label="Added to library",       value=False, style={"marginLeft":"0.25rem","marginBottom":"0.6rem"}),
            dbc.Checkbox(id={"type": "trends-metric", "index": "metacritic"}, label="Metacritic score", value=False, style={"marginLeft":"0.25rem","marginBottom":"0.6rem"}),
            dbc.Checkbox(id={"type": "trends-metric", "index": "youtube_count"}, label="YouTube Views", value=False, style={"marginLeft":"0.25rem","marginBottom":"0.6rem"}),
            dbc.Checkbox(id={"type": "trends-metric", "index": "twitch_count"},  label="Twitch Views",   value=False, style={"marginLeft":"0.25rem","marginBottom":"1rem"}),

            html.Label("Other Options", style={"fontWeight": "bold", "fontSize": "1.1rem", "marginTop": "1rem"}),
            dbc.Checkbox(id="show-trendline-checkbox",     value=False, label="Show trendline", style={"marginLeft":"0.25rem","marginBottom":"0.6rem"}),
            dbc.Checkbox(id="show-genre-average-checkbox", value=False, label="Show averages",   style={"marginLeft":"0.25rem","marginBottom":"0.6rem"}),

            html.Div(
                [
                    html.Label("Select Genre(s)", style={"fontWeight":"bold","marginTop":"1rem"}),
                    dcc.Dropdown(
                        id={"type": "genre-dropdown", "context": "trends"}, options=[], multi=True,
                        placeholder="Select genres...", style={"marginBottom":"1rem"}
                    )
                ]
            ),

            html.Div(
                [
                    html.Label("Select Year Range", style={"fontWeight":"bold","marginTop":"1rem"}),
                    dcc.RangeSlider(
                        id="trends-year-range-slider", min=1970, max=2024, step=1,
                        value=[2000, 2020], marks={y: str(y) for y in range(1970, 2025, 10)},
                        tooltip={"placement":"bottom"}
                    ),

                    html.Label("Number of Games", style={"fontWeight":"bold","marginTop":"1.5rem"}),
                    dcc.Slider(
                        id="trends-num-games-slider",
                        min=1000, max=9000, step=500, value=3000,
                        marks={i: str(i) for i in range(1000,9001,2000)},
                        tooltip={"placement":"bottom"}
                    )
                ],
                style={"marginTop":"1rem"}
            )
        ],
        style={"display": "block" if active_tab == "trends" else "none"}
    )

    compare_button = dbc.Button(
        "Compare Games", id="compare-button", n_clicks=0,
        color="secondary",
        style={"width":"100%", "marginTop":"1rem"}
    )

    theme_toggle = html.Div(
        [
            dbc.Button(
                "🌙 Dark Mode" if theme == "light" else "☀️ Light Mode",
                id="theme-toggle-button",
                n_clicks=0,
                color="secondary",
                style={"width": "100%", "marginTop": "1rem"}
            )
        ]
    )

    return html.Div(
        [
            build_search_bar(theme),
            html.Hr(),
            build_top_control(active_tab),
            html.Hr(),
            html.Div(
                id="middle_options",
                children=[
                    games_controls,
                    games_year_slider,
                    games_num_slider,
                    trends_controls
                ],
                style={"paddingTop":"1rem","paddingBottom":"1rem"}
            ),
            compare_button,
            theme_toggle
        ],
        style={"padding":"1rem"}
    )

# -------------------------
# Layout
# -------------------------
initial_active_tab = "games"

app.layout = html.Div(
    [
        dcc.Store(id="active_main_tab",           data=initial_active_tab),
        dcc.Store(id="selected_sort_options",     data=["Added"]),
        dcc.Store(id="selected_genres",           data=[]),
        dcc.Store(id="selected_year_range",       data=[2000, 2020]),
        dcc.Store(id="selected_game_ids",         data=[]),
        dcc.Store(id="trends_sort_metric",        data="rating"),
        dcc.Store(id="show_trendline",            data=False),
        dcc.Store(id="show_genre_average",        data=False),
        dcc.Store(id="trends_year_range",         data=[2000, 2020]),
        dcc.Store(id="trends_num_games",          data=1000),
        dcc.Store(id="last_clicked_timestamp",    data=None),
        dcc.Store(id="compare_active",            data=False),
        dcc.Store(id="selected_game_id",          data=None),
        dcc.Store(id="theme",                     data="light"),  # Store for theme state

        html.Link(id="theme-stylesheet", rel="stylesheet", href=dbc.themes.BOOTSTRAP),

        html.Div(
            id="root-container",
            children=[
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    id="sidebar",
                                    children=build_sidebar(
                                        initial_active_tab,
                                        ["Added"],
                                        theme="light"
                                    ),
                                    width=3,
                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "height": "100vh",
                                        "padding": 0,
                                        "borderRight": "1px solid #dee2e6",
                                        "display": "flex",
                                        "flexDirection": "column"
                                    }
                                ),
                                dbc.Col(
                                    id="main-content",
                                    children=build_data_view(theme="light"),
                                    width=9,
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "height": "100vh",
                                        "padding": 0
                                    }
                                )
                            ]
                        )
                    ],
                    fluid=True,
                    id="main-container"
                )
            ],
            style={
                "width": "100%",
                "height": "100vh",
                "overflow": "hidden"
            }
        )
    ],
    style={
        "overflow": "auto",
        "height": "100vh",
        "width": "100vw"
    }
)

# =========================
# Callbacks
# =========================

# -- Theme toggle callback ---------------------------------
@app.callback(
    Output("theme", "data"),
    Output("theme-stylesheet", "href"),
    Input("theme-toggle-button", "n_clicks"),
    State("theme", "data")
)
def toggle_theme(n_clicks, current_theme):
    if n_clicks:
        new_theme = "dark" if current_theme == "light" else "light"
        stylesheet = dbc.themes.CYBORG if new_theme == "dark" else dbc.themes.BOOTSTRAP
        return new_theme, stylesheet
    return current_theme, dash.no_update

# -- Update sidebar with theme -----------------------------
@app.callback(
    Output("sidebar", "children"),
    Output("sidebar", "style"),
    [
        Input("active_main_tab", "data"),
        Input("selected_sort_options", "data"),
        Input("theme", "data")
    ]
)
def update_sidebar(active_tab, selected_sort_options, theme):
    sidebar_style = {
        "backgroundColor": "#f8f9fa" if theme == "light" else "#212529",
        "height": "100vh",
        "padding": 0,
        "borderRight": "1px solid #dee2e6" if theme == "light" else "1px solid #495057",
        "display": "flex",
        "flexDirection": "column",
        "color": "#000" if theme == "light" else "#fff"
    }
    return build_sidebar(active_tab, selected_sort_options, theme), sidebar_style

# -- Update main content with theme ------------------------
@app.callback(
    Output("main-content", "children"),
    Output("main-container", "style"),
    Input("theme", "data")
)
def update_main_content(theme):
    container_style = {
        "backgroundColor": "#fff" if theme == "light" else "#121212",
        "color": "#000" if theme == "light" else "#fff"
    }
    return build_data_view(theme), container_style

# -- Main tab / sub-button switching -----------------------
@app.callback(
    Output("active_main_tab", "data"),
    [Input("games-button", "n_clicks"),
     Input("trends-button", "n_clicks")],
    State("active_main_tab", "data")
)
def handle_clicks(games_clicks, trends_clicks, current_tab):
    trig = ctx.triggered_id
    if trig == "games-button":
        return "games"
    if trig == "trends-button":
        return "trends"
    return current_tab

# -- Year range (games tab) --------------------------------
@app.callback(
    Output("selected_year_range", "data"),
    Input("games-year-range-slider", "value"),
    prevent_initial_call=True
)
def update_selected_year_range(year_range):
    return year_range

# -- Trend-tab checkboxes/sliders --------------------------
@app.callback(Output("show_trendline", "data"),
              Input("show-trendline-checkbox", "value"),
              prevent_initial_call=True)
def upd_trendline(v): return v or False

@app.callback(Output("show_genre_average", "data"),
              Input("show-genre-average-checkbox", "value"))
def upd_avg(v): return v or False

@app.callback(Output("trends_year_range", "data"),
              Input("trends-year-range-slider", "value"))
def upd_trends_year(v): return v

@app.callback(Output("trends_num_games", "data"),
              Input("trends-num-games-slider", "value"))
def upd_trends_num(v): return v

# -- Populate genre dropdown (trends tab) ------------------
@app.callback(
    Output({"type": "genre-dropdown", "context": ALL}, "options"),
    Input("active_main_tab", "data")
)
def populate_all_genre_dropdowns(tab):
    genre_set = set()
    for entry in final_df["genres"].dropna():
        if isinstance(entry, list):
            genre_set.update(entry)
        elif isinstance(entry, str):
            genre_set.update([g.strip() for g in entry.split(",")])
    options = [{"label": g, "value": g} for g in sorted(genre_set)]
    return [options] * 2  # games and trends

@app.callback(
    Output("selected_genres", "data"),
    Input({"type": "genre-dropdown", "context": ALL}, "value"),
    State("active_main_tab", "data"),
    prevent_initial_call=True
)
def update_selected_genres_from_dropdown(values, active_tab):
    if active_tab == "games":
        return values[0] if len(values) > 0 else []
    elif active_tab == "trends":
        return values[1] if len(values) > 1 else []
    return []

# -- Enforce single sort selection (games tab) -------------
@app.callback(
    [Output({"type": "sort-button", "index": ALL}, "value"),
     Output("selected_sort_options", "data")],
    Input({"type": "sort-button", "index": ALL}, "value"),
    State({"type": "sort-button", "index": ALL}, "id"),
    prevent_initial_call=True
)
def enforce_single_sort(values, ids):
    trig = ctx.triggered_id
    sel = trig["index"] if trig else next((id_["index"] for v, id_ in zip(values, ids) if v), "Added")
    updated = [id_["index"] == sel for id_ in ids]
    return updated, [sel]

# -- Single metric selection (trends tab) ------------------
@app.callback(
    [Output({"type": "trends-metric", "index": ALL}, "value"),
     Output("trends_sort_metric", "data")],
    Input({"type": "trends-metric", "index": ALL}, "value"),
    State({"type": "trends-metric", "index": ALL}, "id"),
    prevent_initial_call=True
)
def enforce_single_metric(values, ids):
    trig = ctx.triggered_id
    sel = trig["index"] if trig else next((id_["index"] for v, id_ in zip(values, ids) if v), "rating")
    updated = [id_["index"] == sel for id_ in ids]
    return updated, sel

# -- Sidebar search – returns clickable game links --------
@app.callback(
    Output("search_results", "children"),
    Input("search_bar", "value"),
    Input("theme", "data"),
    prevent_initial_call=True
)
def update_search_results(search_text, theme):
    if not search_text or len(search_text.strip()) < 2:
        return ""
    mask = final_df["name"].str.contains(search_text, case=False, na=False)
    matches = final_df.loc[mask].head(10)
    if matches.empty:
        return html.Small("No matches", style={"color": "#888" if theme == "light" else "#ccc"})
    return dbc.ListGroup(
        [
            dbc.ListGroupItem(
                html.Span(
                    row["name"],
                    id={"type": "sidebar-game-link", "index": int(row["id"])},
                    n_clicks=0,
                    style={"color": "#0d6efd", "cursor": "pointer", "textDecoration": "underline"}
                ),
                style={"padding": "0.4rem 0.6rem"}
            )
            for _, row in matches.iterrows()
        ],
        flush=True
    )

# -- Comparison-panel search -------------------------------
@app.callback(
    Output("compare_search_results", "children"),
    Input("compare_search_bar", "value"),
    Input("theme", "data"),
    prevent_initial_call=True
)
def update_compare_search_results(search_text, theme):
    if not search_text or len(search_text.strip()) < 2:
        return ""
    mask = final_df["name"].str.contains(search_text, case=False, na=False)
    matches = final_df.loc[mask].head(10)
    if matches.empty:
        return html.Small("No matches", style={"color": "#888" if theme == "light" else "#ccc"})
    return dbc.ListGroup(
        [
            dbc.ListGroupItem(
                html.Span(
                    row["name"],
                    id={"type": "game-link", "index": int(row["id"])},
                    n_clicks=0,
                    style={
                        "color": "#0d6efd",
                        "cursor": "pointer",
                        "textDecoration": "underline"
                    }
                ),
                style={"padding": "0.4rem 0.6rem"}
            )
            for _, row in matches.iterrows()
        ],
        flush=True
    )

# -- Comparison panel toggle + plot + button color toggle --
@app.callback(
    Output("compare_active", "data"),
    Output("selected_game_ids", "data", allow_duplicate=True),
    Output("comparison_panel", "children"),
    Output("comparison_panel", "style"),
    Output("main_graph", "style"),
    Input("compare-button", "n_clicks"),
    Input("selected_game_ids", "data"),
    State("compare_active", "data"),
    State("theme", "data"),
    prevent_initial_call=True
)
def update_comparison_panel_and_resize(btn_clicks, game_ids, is_active, theme):
    base_main_style = {
        "minHeight": 0, "position": "relative", "width": "100%",
        "height": "100%",  # Ensure it takes full height
        "overflow": "hidden", "transition": "flex 0.3s ease"
    }
    base_compare_style = {
        "overflowY": "auto",
        "borderTop": "1px solid #dee2e6" if theme == "light" else "1px solid #495057",
        "backgroundColor": "#f8f9fa" if theme == "light" else "#212529",
        "transition": "flex 0.3s ease"
    }

    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if trig == "compare-button" and btn_clicks:
        is_active = not is_active

    if not is_active:
        return (
            False, dash.no_update, [],
            base_compare_style | {"display": "none"},
            base_main_style | {"flex": "1 1 0%"}
        )

    selected_df = final_df[final_df["id"].isin(game_ids)]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    # Search bar and icons in same row
    compare_controls_row = html.Div(
        [
            html.Div(
                build_compare_search_bar(theme),
                style={"flex": "0 0 auto"}
            ),
            html.Div(
                [
                    html.Span("Selected games:", style={"fontWeight": "bold", "marginRight": "1rem"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Img(
                                        src=row.get("background_image", ""),
                                        style={
                                            "width": "40px", "height": "40px", "objectFit": "cover",
                                            "borderRadius": "50%",
                                            "border": f"2px solid {colors[idx % len(colors)]}",
                                            "boxShadow": "0 0 3px rgba(0,0,0,0.2)",
                                            "cursor": "default"
                                        },
                                        title=row.get("name", "Unnamed Game")
                                    ),
                                    html.Button(
                                        "×",
                                        id={"type": "remove-game", "index": gid},
                                        style={
                                            "position": "absolute", "top": "-5px", "right": "-5px",
                                            "border": "none", "background": "red", "color": "white",
                                            "borderRadius": "50%", "width": "14px", "height": "14px",
                                            "fontSize": "10px", "lineHeight": "12px", "padding": "0",
                                            "cursor": "pointer"
                                        }
                                    )
                                ],
                                style={"position": "relative", "marginRight": "0.5rem"}
                            )
                            for idx, gid in enumerate(game_ids)
                            for _, row in selected_df[selected_df["id"] == gid].iterrows()
                        ],
                        style={"display": "flex", "gap": "0.5rem"}
                    )
                ],
                style={"display": "flex", "alignItems": "center", "marginLeft": "2rem"}
            )
        ],
        style={"display": "flex", "flexDirection": "row", "alignItems": "center", "padding": "0.5rem 1rem"}
    )

    if not game_ids:
        placeholder = html.Div(
            "Select games to compare.",
            style={"padding": "2rem", "textAlign": "center",
                   "color": "#666" if theme == "light" else "#ccc", "fontSize": "1.2rem"}
        )
        panel_children = html.Div(
            [compare_controls_row, placeholder],
            style={"display": "flex", "flexDirection": "column",
                   "width": "100%", "height": "100%"}
        )
        return (
            True, game_ids, panel_children,
            base_compare_style | {"flex": "1.4 1 0%", "display": "block"},
            base_main_style | {"flex": "2 1 0%"}
        )

    metric_max = {
        "rating": 5.0,
        "metacritic": 100.0,
        "added": selected_df["added"].max() or 1,
        "youtube_count": selected_df["youtube_count"].max() or 1,
        "twitch_count": selected_df["twitch_count"].max() or 1
    }

    labels = ["Rating", "Metacritic", "Added", "YouTube", "Twitch"]
    fig = go.Figure()

    for idx, gid in enumerate(game_ids):
        row = selected_df[selected_df["id"] == gid].iloc[0]
        raw = [row.get("rating", 0), row.get("metacritic", 0), row.get("added", 0),
               row.get("youtube_count", 0), row.get("twitch_count", 0)]
        norm = [raw[0]/metric_max["rating"], raw[1]/metric_max["metacritic"],
                raw[2]/metric_max["added"], raw[3]/metric_max["youtube_count"],
                raw[4]/metric_max["twitch_count"]]
        fig.add_trace(
            go.Bar(
                x=labels, y=norm,
                name=row.get("name", f"Game {idx+1}"),
                marker_color=colors[idx % len(colors)],
                marker_line_width=0, opacity=1.0,
                customdata=[f"{v:.1f}" if isinstance(v, (int, float)) else str(v) for v in raw],
                hovertemplate="%{customdata}<extra></extra>"
            )
        )

    fig.update_layout(
        dragmode="pan",
        barmode="group",
        xaxis=dict(showline=False, showgrid=False, ticks="", zeroline=False,
                   tickfont=dict(size=12)),
        yaxis=dict(visible=False, range=[0, 1.05], fixedrange=True),
        bargap=0.3, bargroupgap=0.1,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#fff" if theme == "light" else "#212529",
        paper_bgcolor="#fff" if theme == "light" else "#212529",
        font=dict(color="#000" if theme == "light" else "#fff"),
        showlegend=False
    )

    histogram = html.Div(
        dcc.Graph(figure=fig, 
                  config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": [
                        "toImage", "sendDataToCloud", "editInChartStudio",
                        "hoverClosestCartesian", "hoverCompareCartesian",
                        "toggleSpikelines", "select2d", "lasso2d"
                    ],
                    "displaylogo": False
                    },
                style={"height": "100%", "width": "100%"}),
        style={"flex": "1", "padding": "0.5rem",
               "overflow": "hidden", "display": "flex", "alignItems": "center"}
    )

    row = html.Div(
        [histogram],
        style={"display": "flex", "flexDirection": "row",
               "width": "100%", "height": "100%", "flex": "1 1 0"}
    )

    panel_children = html.Div(
        [compare_controls_row, row],
        style={"display": "flex", "flexDirection": "column",
               "width": "100%", "height": "100%"}
    )

    return (
        True, game_ids, panel_children,
        base_compare_style | {"flex": "1.4 1 0%", "display": "block"},
        base_main_style | {"flex": "2 1 0%"}
    )

@app.callback(
    Output("selected_game_ids", "data"),
    Output("last_clicked_timestamp", "data"),
    Output("selected_game_id", "data"),
    Input({"type": "game-link", "index": ALL}, "n_clicks_timestamp"),
    Input({"type": "remove-game", "index": ALL}, "n_clicks"),
    Input({"type": "game-tile", "index": ALL}, "n_clicks"),
    State({"type": "game-link", "index": ALL}, "id"),
    State("selected_game_ids", "data"),
    State("last_clicked_timestamp", "data"),
    State({"type": "game-tile", "index": ALL}, "id"),
    prevent_initial_call=True
)
def unified_game_selection(
    game_link_clicks, remove_clicks, tile_clicks,
    game_link_ids, selected_ids, last_ts, tile_ids
):
    triggered = ctx.triggered_id
    if selected_ids is None:
        selected_ids = []

    # Handle REMOVE click
    if isinstance(triggered, dict) and triggered.get("type") == "remove-game":
        return [gid for gid in selected_ids if gid != triggered["index"]], last_ts, dash.no_update

    # Handle TILE click
    if isinstance(triggered, dict) and triggered.get("type") == "game-tile":
        return selected_ids, last_ts, triggered["index"]

    # Handle LINK click from search
    max_ts, clicked = -1, None
    for ts, btn_id in zip(game_link_clicks, game_link_ids):
        if ts and ts > max_ts:
            max_ts, clicked = ts, btn_id["index"]

    if clicked and (last_ts is None or max_ts > last_ts):
        if clicked not in selected_ids:
            if len(selected_ids) == 5:
                selected_ids.pop(0)
            selected_ids.append(clicked)
        return selected_ids, max_ts, clicked

    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("selected-game-icons", "children"),
    Input("selected_game_ids", "data")
)
def update_selected_game_icons(game_ids):
    if not game_ids:
        return []
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    return generate_selected_game_icons(game_ids, final_df, colors)

@app.callback(
    [Output("game-details-panel", "children"),
     Output("game-details-panel", "style")],
    Input("selected_game_id", "data"),
    State("compare_active", "data"),
    State("theme", "data"),
    prevent_initial_call=True
)
def update_game_details_panel(game_id, compare_active, theme):
    if not game_id or compare_active:
        return (
            html.Div("Select a game to view details.", style={"color": "#888" if theme == "light" else "#ccc", "padding": "1rem"}),
            {
                "position": "absolute",
                "top": 0,
                "right": 0,
                "width": "360px",
                "height": "100vh",
                "backgroundColor": "#f8f9fa" if theme == "light" else "#212529",
                "borderLeft": "1px solid #dee2e6" if theme == "light" else "1px solid #495057",
                "overflowY": "auto",
                "boxShadow": "-2px 0 8px rgba(0,0,0,0.15)",
                "transition": "transform 0.3s ease-in-out",
                "transform": "translateX(100%)",
                "zIndex": 1000,
                "padding": "1rem"
            }
        )

    match = final_df[final_df["id"] == game_id]
    if match.empty:
        return (
            html.Div("Game not found.", style={"color": "#888" if theme == "light" else "#ccc", "padding": "1rem"}),
            {
                "position": "absolute",
                "top": 0,
                "right": 0,
                "width": "360px",
                "height": "100vh",
                "backgroundColor": "#f8f9fa" if theme == "light" else "#212529",
                "borderLeft": "1px solid #dee2e6" if theme == "light" else "1px solid #495057",
                "overflowY": "auto",
                "boxShadow": "-2px 0 8px rgba(0,0,0,0.15)",
                "transition": "transform 0.3s ease-in-out",
                "transform": "translateX(100%)",
                "zIndex": 1000,
                "padding": "1rem"
            }
        )

    row = match.iloc[0]
    panel_content = dbc.Card([
        dbc.CardHeader(
            [
                html.Span(row.get("name", "Unnamed Game"), style={"fontWeight": "bold", "fontSize": "1.5rem"}),
                html.Button(
                    "×",
                    id="close-game-panel-button",
                    n_clicks=0,
                    style={
                        "position": "absolute",
                        "top": "10px",
                        "right": "10px",
                        "fontSize": "20px",
                        "background": "none",
                        "border": "none",
                        "color": "#888" if theme == "light" else "#ccc",
                        "cursor": "pointer"
                    }
                )
            ]
        ),
        html.Div([
            html.Img(
                src=row.get("background_image", ""),
                style={"width": "100%", "height": "200px", "objectFit": "cover", "borderRadius": "8px"}
            ) if row.get("background_image") else None,
            html.Hr(),
            html.P(f"Player Rating: {row.get('rating', 'N/A')}"),
            html.P(f"Metacritic: {row.get('metacritic', 'N/A')}"),
            html.P(f"Released: {row.get('released', 'Unknown')}"),
            html.P(f"Added by users: {row.get('added', 'N/A')}"),
            html.P(f"YouTube count: {row.get('youtube_count', 'N/A')}"),
            html.P(f"Twitch count: {row.get('twitch_count', 'N/A')}"),
            html.P(
                (row.get("description_raw") or "") + "..." if isinstance(row.get("description_raw"), str) else "No description available.",
                style={"fontSize": "0.9rem"}
            )
        ], style={"padding": "1rem"})
    ], style={"marginBottom": "1rem", "position": "relative"})

    panel_style = {
        "position": "absolute",
        "top": 0,
        "right": 0,
        "width": "360px",
        "height": "100vh",
        "backgroundColor": "#f8f9fa" if theme == "light" else "#212529",
        "borderLeft": "1px solid #dee2e6" if theme == "light" else "1px solid #495057",
        "overflowY": "auto",
        "boxShadow": "-2px 0 8px rgba(0,0,0,0.15)",
        "transition": "transform 0.3s ease-in-out",
        "transform": "translateX(0%)",
        "zIndex": 1000,
        "padding": "1rem"
    }

    return panel_content, panel_style

@app.callback(
    Output("selected_game_id", "data", allow_duplicate=True),
    Input("close-game-panel-button", "n_clicks"),
    prevent_initial_call=True
)
def close_game_panel(n_clicks):
    if n_clicks:
        return None
    return dash.no_update

# -- MAIN GRAPH --------------------------------------------
@app.callback(
    Output("games_graph", "children", allow_duplicate=True),
    [
        Input("selected_sort_options", "data"),
        Input("selected_year_range", "data"),
        Input("games-num-games-slider", "value"),
        Input("selected_genres", "data"),
        Input("theme", "data")
    ],
    State("active_main_tab", "data"),
    prevent_initial_call=True
)
def update_main_graph_games(
    selected_sort_options,
    selected_year_range,
    num_games,
    selected_genres,
    theme,
    active_tab,
):
    if active_tab != "games":
        raise PreventUpdate

    df = final_df.copy().dropna(subset=["name"]).drop_duplicates()
    df["release_year"] = pd.to_datetime(df["released"], errors="coerce").dt.year
    df = df[
        (df["release_year"] >= selected_year_range[0]) &
        (df["release_year"] <= selected_year_range[1])
        ]

    mapping = {
        "Player Rating":     "rating",
        "YouTube Views":    "youtube_count",
        "Twitch Views":     "twitch_count",
        "Added to library":      "added",
        "Metacritic score": "metacritic"
    }

    display_labels = {
        "rating": "Player Rating",
        "youtube_count": "YouTube Views",
        "twitch_count": "Twitch Mentions",
        "added": "Library Additions",
        "metacritic": "Metacritic Score"
    }

    sort_key = (selected_sort_options[0] if selected_sort_options else "Added to library")
    sort_by = mapping.get(sort_key, "added")
    title_label = display_labels.get(sort_by, sort_key)

    if selected_genres:
        def has_matching_genre(genre_str):
            try:
                for g in [g_.strip() for g_ in genre_str.split(",")]:
                    if g in selected_genres:
                        return True
            except Exception:
                pass
            return False
        df = df[df["genres"].apply(has_matching_genre)]

    df = (df.dropna(subset=[sort_by])
            .sort_values(sort_by, ascending=False)
            .iloc[num_games[0]-1 : num_games[1]])

    metric = df[sort_by].astype(float).clip(lower=1e-6)
    container_width = 3  # Approximate relative units
    container_height = 2

    # Normalize sizes to fit the container's aspect ratio
    normed = squarify.normalize_sizes(metric, container_width * 100, container_height * 100)
    rects = squarify.squarify(normed, 0, 0, container_width * 100, container_height * 100)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rects)], axis=1)

    tiles = [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            f"{i}. {row['name']}",
                            style={
                                "fontSize": "12px", "fontWeight": "bold",
                                "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"
                            }
                        ),
                        html.Div(
                            f"{sort_by.capitalize()}: {row[sort_by]:.2f}",
                            style={"fontSize": "10px"}
                        )
                    ],
                    style={
                        "position": "absolute", "inset": 0,
                        "backgroundColor": "rgba(0,0,0,0.55)",
                        "display": "flex", "flexDirection": "column",
                        "alignItems": "center", "justifyContent": "center",
                        "padding": "4px",
                    }
                )
            ],
            id={"type": "game-tile", "index": int(row["id"])},
            n_clicks=0,
            style={
                "position": "absolute",
                "left": f"{(row['x'] / (container_width * 100)) * 100:.2f}%",
                "top": f"{(row['y'] / (container_height * 100)) * 100:.2f}%",
                "width": f"{(row['dx'] / (container_width * 100)) * 100:.2f}%",
                "height": f"{(row['dy'] / (container_height * 100)) * 100:.2f}%",
                "backgroundImage": f"url('{row['background_image']}')",
                "backgroundSize": "cover", "backgroundPosition": "center",
                "border": "1px solid #fff", "boxSizing": "border-box",
                "borderRadius": "4px", "overflow": "hidden", "color": "#fff",
                "display": "flex", "justifyContent": "center", "alignItems": "center",
                "textAlign": "center", "flexDirection": "column", "padding": "4px",
                "cursor": "pointer"
            }
        )
        for i , (_, row) in enumerate(df.iterrows(), start=num_games[0])
    ]

    return html.Div(
        [
            html.Div(
                f"Top Games by {title_label}",
                style={
                    "padding": "0.5rem 1rem",
                    "fontWeight": "bold",
                    "fontSize": "1.2rem",
                    "color": "#000" if theme == "light" else "#fff",
                    "backgroundColor": "#f1f3f5" if theme == "light" else "#1a1a1a"
                }
            ),
            html.Div(tiles, style={
                "position": "relative",
                "width": "100%",
                "height": "100%",
                "backgroundColor": "#333" if theme == "light" else "#1a1a1a"
            })
        ],
        style={"height": "100%", "width": "100%"}
    )

@app.callback(
    Output("trends_graph", "children", allow_duplicate=True),
    [
        Input("trends_year_range", "data"),
        Input("trends_num_games", "data"),
        Input("trends_sort_metric", "data"),
        Input("show_trendline", "data"),
        Input("show_genre_average", "data"),
        Input("selected_genres", "data"),
        Input("theme", "data")
    ],
    State("active_main_tab", "data"),
    prevent_initial_call=True
)
def update_main_graph_trends(
    trends_year_range,
    trends_num_games,
    trends_sort_metric,
    show_trendline,
    show_genre_average,
    selected_genres,
    theme,
    active_tab,
):
    if active_tab != "trends":
        raise PreventUpdate

    df = final_df.copy().dropna(subset=["name"]).drop_duplicates()
    df["release_year"] = pd.to_datetime(df["released"], errors="coerce").dt.year

    df = df[
        (df["release_year"] >= trends_year_range[0]) &
        (df["release_year"] <= trends_year_range[1])
    ].dropna(subset=["released", "rating", "genres"])
    df["released"] = pd.to_datetime(df["released"], errors="coerce")
    df = df[df["rating"] > 0.5]

    if selected_genres:
        def extract_matching_genre(genre_str):
            try:
                for g in [g_.strip() for g_ in genre_str.split(",")]:
                    if g in selected_genres:
                        return g
            except Exception:
                pass
            return None
        df["matched_genre"] = df["genres"].apply(extract_matching_genre)
        df = df[df["matched_genre"].notna()]
        color_col = "matched_genre"
    else:
        df["matched_genre"] = "All"
        color_col = None

    df = df.sample(n=min(trends_num_games, len(df)), random_state=42)
    df = df.sort_values("released")

    y_metric = trends_sort_metric.lower()
    y_label = {
        "rating": "Player Rating",
        "added": "Added by Users",
        "metacritic": "Metacritic Score",
        "youtube_count": "YouTube Mentions",
        "twitch_count": "Twitch Mentions"
    }.get(y_metric, y_metric.capitalize())

    df[y_metric] = pd.to_numeric(df[y_metric], errors="coerce")
    df = df.dropna(subset=["released", y_metric])

    x_ord = df["released"].map(pd.Timestamp.toordinal)
    y_val = df[y_metric]

    fig = go.Figure()

    if show_genre_average:
        df["year"] = df["released"].dt.year
        grouped = df.groupby(["year", "matched_genre"])[y_metric].mean().reset_index()
        for genre in grouped["matched_genre"].unique():
            gdf = grouped[grouped["matched_genre"] == genre]
            fig.add_trace(go.Scatter(
                x=gdf["year"], y=gdf[y_metric],
                mode="lines+markers", name=genre,
                marker=dict(size=6, opacity=0.8),
                line=dict(width=2)
            ))
    else:
        for genre in df["matched_genre"].unique():
            gdf = df[df["matched_genre"] == genre]
            fig.add_trace(go.Scatter(
                x=gdf["released"], y=gdf[y_metric],
                mode="markers", name=genre,
                text=gdf["name"],
                marker=dict(size=6, opacity=0.6)
            ))

    if show_trendline and len(x_ord) > 1:
        X = sm.add_constant(x_ord)
        model = sm.OLS(y_val, X).fit()
        trend_y = model.predict(X)
        fig.add_trace(go.Scatter(
            x=df["released"], y=trend_y,
            mode="lines", name="Trendline",
            line=dict(color="orange", width=2)
        ))

    y_min, y_max = y_val.min(), y_val.max()
    padding = (y_max - y_min) * 0.05 if y_max > y_min else 1
    fig.update_layout(
        dragmode="pan",
        title=f"Game {y_label} Over Time" + (" by Genre" if selected_genres else ""),
        xaxis_title="Release Date",
        yaxis_title=y_label,
        yaxis_range=[max(0, y_min - padding), y_max + padding],
        transition_duration=500,
        plot_bgcolor="#fff" if theme == "light" else "#212529",
        paper_bgcolor="#fff" if theme == "light" else "#212529",
        font=dict(color="#000" if theme == "light" else "#fff")
    )

    return dcc.Graph(figure=fig,
                     config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": [
                        "toImage", "sendDataToCloud", "editInChartStudio",
                        "hoverClosestCartesian", "hoverCompareCartesian",
                        "toggleSpikelines", "select2d", "lasso2d"
                    ],
                    "displaylogo": False
                    },
                     style={"height": "100%", "width": "100%"})

@app.callback(
    Output("games_graph", "children", allow_duplicate=True),
    Output("trends_graph", "children", allow_duplicate=True),
    Input("active_main_tab", "data"),
    Input("theme", "data"),
    prevent_initial_call=True
)
def reset_graphs_on_tab_switch(tab, theme):
    if tab == "games":
        return build_loading_component("Loading games...", theme), []
    elif tab == "trends":
        return [], build_loading_component("Loading trends...", theme)
    return [], []

@app.callback(
    Output("games_graph", "style"),
    Output("trends_graph", "style"),
    Input("active_main_tab", "data"),
)
def toggle_graph_visibility(tab):
    base_style = {"height": "100%", "width": "100%"}  # Ensure full height for both graphs
    if tab == "games":
        return {"display": "block", **base_style}, {"display": "none"}
    if tab == "trends":
        return {"display": "none"}, {"display": "block", **base_style}
    return {"display": "none"}, {"display": "none"}

@app.callback(
    Output("selected_game_id", "data", allow_duplicate=True),
    Output("last_clicked_timestamp", "data", allow_duplicate=True),
    Input({"type": "sidebar-game-link", "index": ALL}, "n_clicks_timestamp"),
    State({"type": "sidebar-game-link", "index": ALL}, "id"),
    State("last_clicked_timestamp", "data"),
    prevent_initial_call=True
)
def handle_sidebar_game_link_click(timestamps, ids, last_ts):
    max_ts, clicked = -1, None
    for ts, btn_id in zip(timestamps, ids):
        if ts and ts > max_ts:
            max_ts = ts
            clicked = btn_id["index"]


    if clicked and (last_ts is None or max_ts > last_ts):
        return clicked, max_ts

    return dash.no_update, dash.no_update

@app.callback(
    Output("selected_game_ids", "data", allow_duplicate=True),
    Output("last_clicked_timestamp", "data", allow_duplicate=True),
    Output("selected_game_id", "data", allow_duplicate=True),
    Input({"type": "compare-game-link", "index": ALL}, "n_clicks_timestamp"),
    State({"type": "compare-game-link", "index": ALL}, "id"),
    State("selected_game_ids", "data"),
    State("last_clicked_timestamp", "data"),
    State("selected_game_id", "data"),
    prevent_initial_call=True
)
def handle_compare_game_link_click(timestamps, ids, selected_ids, last_ts, selected_game_id):
    if selected_ids is None:
        selected_ids = []

    max_ts, clicked = -1, None
    for ts, btn_id in zip(timestamps, ids):
        if ts and ts > max_ts:
            max_ts = ts
            clicked = btn_id["index"]

    if clicked and (last_ts is None or max_ts > last_ts):
        if clicked not in selected_ids:
            if len(selected_ids) >= 5:
                selected_ids.pop(0)
            selected_ids.append(clicked)
        return selected_ids, max_ts, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update

# =========================
# Run server (dynamic port)
# =========================
import socket

def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

PORT = 6969

def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{PORT}")

if __name__ == "__main__":
    # threading.Timer(1, open_browser).start()
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=PORT)


# In[ ]:




