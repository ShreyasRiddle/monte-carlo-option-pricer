import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np

from monte_carlo import generate_gbm_paths, price_option_mc
from black_scholes import black_scholes_price

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Monte Carlo Option Pricer"

input_style = {
    "backgroundColor": "#0f0f0f",
    "color": "#ffffff",
    "border": "none"
}

app.layout = html.Div(
    style={"backgroundColor": "#0f0f0f", "padding": "20px"},
    children=[
        html.H2("Monte Carlo Option Pricer", style={"color": "white", "marginBottom": "30px"}),

        dbc.Row([
            # ---- Left column: controls ----
            dbc.Col(html.Div(
                style={"backgroundColor": "#1e1e1e","padding": "20px","borderRadius": "8px"},
                children=[
                    html.Label("Initial Stock Price (S₀)", style={"color": "white"}),
                    dcc.Input(id='s0', type='number', value=100, step=1,
                              style=input_style, className="form-control mb-3"),

                    html.Label("Strike Price (K)", style={"color": "white"}),
                    dcc.Input(id='K', type='number', value=100, step=1,
                              style=input_style, className="form-control mb-3"),

                    html.Label("Risk-Free Rate (r)", style={"color": "white"}),
                    dcc.Input(id='r', type='number', value=0.05, step=0.01,
                              style=input_style, className="form-control mb-3"),

                    html.Label("Volatility (σ)", style={"color": "white"}),
                    dcc.Input(id='sigma', type='number', value=0.2, step=0.01,
                              style=input_style, className="form-control mb-3"),

                    html.Label("Time to Maturity (T, in years)", style={"color": "white"}),
                    dcc.Input(id='T', type='number', value=1, step=0.1,
                              style=input_style, className="form-control mb-3"),

                    html.Label("Steps per Simulation", style={"color": "white"}),
                    dcc.Input(id='n_steps', type='number', value=252, step=1,
                              style=input_style, className="form-control mb-3"),

                    html.Label("Number of Simulations", style={"color": "white"}),
                    dcc.Input(id='n_sims', type='number', value=500, step=50,
                              style=input_style, className="form-control mb-3"),

                    html.Label("Option Type", style={"color": "white"}),
                    dcc.Dropdown(
                        id='option-type',
                        options=[
                            {'label': 'Call Option', 'value': 'call'},
                            {'label': 'Put Option',  'value': 'put'}
                        ],
                        value='call',
                        className="mb-3",
                        persistence=True,
                        persistence_type="session"
                    ),

                    dbc.Button("Run Simulation", id='run-button',
                               color="primary", className="w-100")
                ]),
                width=4
            ),

            # ---- Right column: outputs ----
            dbc.Col(html.Div(
                style={"backgroundColor": "#1e1e1e","padding": "20px","borderRadius": "8px"},
                children=[
                    html.Div(id='option-price-output',
                             style={"color": "white", "marginBottom": "20px"}),
                    dcc.Graph(id='gbm-graph'),
                    dcc.Graph(id='histogram'),
                    dcc.Graph(id='payoff-histogram'),
                ]),
                width=8
            )
        ])
    ]
)

@app.callback(
    [
        Output('gbm-graph',        'figure'),
        Output('option-price-output','children'),
        Output('histogram',         'figure'),
        Output('payoff-histogram',  'figure'),
    ],
    [
        Input('run-button',  'n_clicks'),
        Input('s0',          'value'),
        Input('K',           'value'),
        Input('r',           'value'),
        Input('sigma',       'value'),
        Input('T',           'value'),
        Input('n_steps',     'value'),
        Input('n_sims',      'value'),
        Input('option-type', 'value'),
    ]
)
def update_output(n_clicks, s0, K, r, sigma, T, n_steps, n_sims, option_type):
    # 1) simulate GBM
    paths = generate_gbm_paths(s0, r, sigma, T, n_steps, n_sims)

    # 2) compute prices
    mc_price = price_option_mc(paths, K, r, T, option_type)
    bs_price = black_scholes_price(s0, K, r, T, sigma, option_type)
    percent_error = abs(mc_price - bs_price) / bs_price * 100

    # ---- GBM paths in unique HSL colors ----
    x_vals = list(range(n_steps + 1))
    fig = go.Figure()
    for i in range(n_sims):
        # hue cycles from 0→360
        hue = (i * 360 / n_sims) 
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=paths[i],
            mode='lines',
            line=dict(color=f"hsl({hue:.1f},70%,50%)", width=1),
            showlegend=False,
            opacity=0.8
        ))

    fig.update_layout(
        title="Simulated GBM Paths",
        xaxis=dict(title="Time Step", color='white'),
        yaxis=dict(title="Price",     color='white'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white')
    )

    # ---- Terminal price histogram ----
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=paths[:, -1],
        nbinsx=40,
        marker=dict(color='white', line=dict(color='#333', width=1))
    ))
    hist_fig.update_layout(
        title="Terminal Stock Price Distribution",
        xaxis=dict(title="S_T", color='white'),
        yaxis=dict(title="Frequency", color='white'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white')
    )

    # ---- Payoff histogram ----
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    else:
        payoffs = np.maximum(K - paths[:, -1], 0)

    payoff_fig = go.Figure()
    payoff_fig.add_trace(go.Histogram(
        x=payoffs,
        nbinsx=40,
        marker=dict(color='white', line=dict(color='#333', width=1))
    ))
    payoff_fig.update_layout(
        title="Payoff Distribution",
        xaxis=dict(title="Payoff", color='white'),
        yaxis=dict(title="Frequency", color='white'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white')
    )

    # ---- Text output ----
    message = html.Span(
        "✔ Monte Carlo estimate is reasonably close to Black-Scholes"
        if percent_error < 5 else
        "⚠ Monte Carlo estimate diverges from Black-Scholes",
        style={"color": "lightgreen" if percent_error < 5 else "orange"}
    )
    price_display = html.Div([
        html.P(f"Monte Carlo {option_type.capitalize()} Price: ${mc_price:.2f}", style={"color":"white"}),
        html.P(f"Black-Scholes Price: ${bs_price:.2f}",         style={"color":"white"}),
        html.P(f"Percent Error: {percent_error:.2f}%",         style={"color":"lightgray"}),
        message
    ])

    return fig, price_display, hist_fig, payoff_fig

if __name__ == '__main__':
    app.run(debug=True)
