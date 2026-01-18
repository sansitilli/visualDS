import os
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "combined_dataset.csv"))

def lock_height(fig, h=520):
    fig.update_layout(
        height=h,
        autosize=False
    )
    return fig

def classify_income(gdp):
    if pd.isna(gdp):
        return None
    if gdp >= 13000:
        return "High Income"
    elif gdp >= 5000:
        return "Upper-Middle Income"
    elif gdp >= 1025:
        return "Lower-Middle Income"
    else:
        return "Low Income"


df["income_group"] = df["gdp_pc"].apply(classify_income)
df = df[df["gdp_pc"] > 0].copy()
df["log_gdp"] = np.log10(df["gdp_pc"])

order = ["High Income", "Upper-Middle Income", "Lower-Middle Income", "Low Income"]
df["income_group"] = pd.Categorical(df["income_group"], categories=order, ordered=True)

needed = [
    "iso3", "country_name",
    "fertility_rate", "log_gdp",
    "female_lfp", "unpaid_work_hours",
    "gender_wage_gap", "women_managers",
    "income_group"
]
df = df[needed].copy()



income_palette = {
    "Low Income": "#FDE725",
    "Lower-Middle Income": "#21918C",
    "Upper-Middle Income": "#3B528B",
    "High Income": "#440154"
}



def bubble_sizes(series: pd.Series) -> pd.Series:
    u_raw = pd.to_numeric(series, errors="coerce")
    u_nonmissing = u_raw.dropna()

    if u_nonmissing.empty:
        return pd.Series(220.0, index=series.index)

    med = float(u_nonmissing.median())
    lo, hi = u_nonmissing.quantile([0.05, 0.95])

    u = u_raw.fillna(med).clip(lo, hi)
    r = u.rank(pct=True)


    return 60 + (r ** 2.0) * 1800


profile_cols = [
    "female_lfp",
    "unpaid_work_hours",
    "gender_wage_gap",
    "women_managers",
    "log_gdp",
    "fertility_rate"
]




def fig_bubble(d: pd.DataFrame):
    dd = d.dropna(subset=["fertility_rate", "log_gdp", "income_group"]).copy()
    dd["size"] = bubble_sizes(dd["unpaid_work_hours"])

    fig = px.scatter(
        dd,
        x="log_gdp",
        y="fertility_rate",
        color="income_group",
        category_orders={"income_group": order}, 
        color_discrete_map=income_palette,
        size="size",
        size_max=60,
        hover_name="country_name",
        custom_data=["iso3"],  
        hover_data={
            "iso3": True,
            "unpaid_work_hours": True,
            "gender_wage_gap": True,
            "female_lfp": True,
            "women_managers": True,
            "log_gdp": ":.2f"
        },
    )

    fig.update_traces(marker=dict(line=dict(width=0.7, color="white"), opacity=0.88))
    fig.update_layout(
        title="Fertility vs development (bubble size = unpaid care burden), 2018",
        xaxis_title="GDP per capita (log10 USD)",
        yaxis_title="Fertility rate (births per woman)",
        legend_title="Income group",
        margin=dict(l=40, r=20, t=60, b=40),
        dragmode="lasso",
        legend_traceorder="normal"
    )
    fig = lock_height(fig, 600)
    return fig



def fig_mechanism(d: pd.DataFrame, xvar: str):
    dd = d.dropna(subset=[xvar, "fertility_rate", "income_group"]).copy()

    fig = px.scatter(
        dd,
        x=xvar,
        y="fertility_rate",
        color="income_group",
        category_orders={"income_group": order}, 
        color_discrete_map=income_palette,
        hover_name="country_name",
        hover_data={"iso3": True}
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=0.7, color="white"), opacity=0.88))

    nice = {
        "unpaid_work_hours": "Unpaid work (hours/day)",
        "gender_wage_gap": "Gender wage gap (%)",
        "female_lfp": "Female labour-force participation (%)",
        "women_managers": "Women in management (%)"
    }

    fig.update_layout(
        title=f"{nice.get(xvar, xvar)} vs fertility (linked selection)",
        xaxis_title=nice.get(xvar, xvar),
        yaxis_title="Fertility rate (births per woman)",
        legend_title="Income group",
        margin=dict(l=40, r=20, t=60, b=40),
        legend_traceorder="normal"
    )
    fig = lock_height(fig, 560)
    dcc.Graph(id="mechanism", style={"height": "560px"})

    return fig



def fig_autonomy_bars(d_all: pd.DataFrame, d_sel: pd.DataFrame):
    cols = profile_cols

    all_num = d_all.dropna(subset=cols).copy()
    sel_num = d_sel.dropna(subset=cols).copy()

    if len(all_num) < 3 or len(sel_num) < 2:
        return go.Figure()

    # standardise on ALL countries
    scaler = StandardScaler().fit(all_num[cols].values)
    all_z = scaler.transform(all_num[cols].values)
    sel_z = scaler.transform(sel_num[cols].values)

    diff = sel_z.mean(axis=0) - all_z.mean(axis=0)

    labels = [
        "Female labour participation",
        "Unpaid care burden",
        "Gender wage gap",
        "Women in management",
        "Economic development",
        "Fertility"
    ]

    colors = ["#2ca02c" if v > 0 else "#1f77b4" for v in diff]

    fig = go.Figure(
        go.Bar(
            x=diff,
            y=labels,
            orientation="h",
            marker_color=colors
        )
    )

    fig.update_layout(
        title="How selected countries differ from the global average (z-scores)",
        xaxis_title="Difference from global mean (standard deviations)",
        yaxis_title="",
        xaxis=dict(zeroline=True, zerolinewidth=2),
        margin=dict(l=60, r=30, t=60, b=40),
        height=420,
        autosize=False
    )

    return fig



def fig_distribution(d_all: pd.DataFrame, d_sel: pd.DataFrame):
    base = d_all.dropna(subset=["fertility_rate", "income_group"]).copy()

    fig = px.box(
        base,
        x="income_group",
        y="fertility_rate",
        category_orders={"income_group": order},
        color="income_group",
        color_discrete_map=income_palette,
        points=False
    )

    fig.update_layout(
        title="Fertility distribution by income group (overlay selection)",
        xaxis_title="Income group",
        yaxis_title="Fertility rate (births per woman)",
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    if len(d_sel) > 0:
        sel = d_sel.dropna(subset=["fertility_rate", "income_group"]).copy()
        fig.add_trace(
            go.Scatter(
                x=sel["income_group"].astype(str),
                y=sel["fertility_rate"],
                mode="markers",
                marker=dict(size=9, color="black", line=dict(width=0.6, color="white")),
                name="Selected"
            )
        )

    fig = lock_height(fig, 560)
    return fig




app = Dash(__name__)
server = app.server  


app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "system-ui"},
    children=[
        html.H2("Fertility & Women’s Autonomy Dashboard (2018)"),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "14px"},
            children=[
                dcc.Graph(
                     id="bubble",
                        figure=fig_bubble(df),
                        clear_on_unhover=True,
                        style={"height": "560px", "gridColumn": "1 / span 2"},
                        config={"responsive": False}
                    ),
                html.Div([
                    html.Div("Choose linked x-variable:", style={"marginBottom": "6px"}),
                    dcc.Dropdown(
                        id="xvar",
                        options=[
                            {"label": "Female labour participation", "value": "female_lfp"},
                            {"label": "Unpaid care burden", "value": "unpaid_work_hours"},
                            {"label": "Gender wage gap", "value": "gender_wage_gap"},
                            {"label": "Women in management", "value": "women_managers"},
                        ],
                        value="unpaid_work_hours",
                        clearable=False
                    ),
                    dcc.Graph(id="mechanism", figure=fig_mechanism(df, "unpaid_work_hours"),
    clear_on_unhover=True,
    style={"height": "560px"}, config={"responsive": False})
                ])
            ]
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px", "marginTop": "10px"},
            children=[
                dcc.Graph(id="profile", figure=fig_autonomy_bars(df, df),
    clear_on_unhover=True,
    style={"height": "420px"},config={"responsive": False}),
                dcc.Graph(id="dist",figure=fig_distribution(df, df),
    clear_on_unhover=True,
    style={"height": "560px"}, config={"responsive": False})
            ]
        ),

        dcc.Store(id="selected_iso3", data=[]),

        html.Div(
            style={"marginTop": "6px", "fontSize": "13px", "opacity": 0.85},
            children="Tip: use lasso/box select on the bubble chart; selection updates the other three views. Double-click to reset."
        )
    ]
)



@app.callback(
    Output("selected_iso3", "data"),
    Input("bubble", "selectedData")
)
def store_selection(selected):
    if not selected or "points" not in selected:
        return []

    iso3 = []
    for p in selected["points"]:
        cd = p.get("customdata")
        if cd and len(cd) > 0 and cd[0]:
            iso3.append(cd[0])

    return sorted(set(iso3))



@app.callback(
    Output("mechanism", "figure"),
    Output("profile", "figure"),
    Output("dist", "figure"),
    Input("xvar", "value"),
    Input("selected_iso3", "data")
)
def update_views(xvar, selected_iso3):
    if selected_iso3:
        d_sel = df[df["iso3"].isin(selected_iso3)].copy()
        if d_sel.empty:
            d_sel = df.copy()
    else:
        d_sel = df.copy()

    return (
        fig_mechanism(d_sel, xvar),
        fig_autonomy_bars(df, d_sel),
        fig_distribution(df, d_sel)
    )


if __name__ == "__main__":
    app.run_server(debug=True)