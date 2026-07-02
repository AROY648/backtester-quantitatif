"""Tous les graphiques Plotly de l'application.

Chaque fonction retourne une figure — l'affichage (st.plotly_chart)
reste dans app.py.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_results(df: pd.DataFrame, buy_signals, sell_signals, ticker: str, metrics: dict):
    """Graphique principal : prix + SMA + signaux, portefeuille vs B&H, drawdown."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.3, 0.2],
        vertical_spacing=0.06,
        subplot_titles=(
            f"{ticker} — Prix et moyennes mobiles",
            "Valeur du portefeuille vs Buy & Hold",
            "Drawdown"
        )
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Prix",
        line=dict(color="#378ADD", width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>Prix: $%{y:.2f}<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA_fast"],
        name=f"SMA {df.attrs.get('short_w', 'rapide')}",
        line=dict(color="#EF9F27", width=1.5, dash="dot"),
        hovertemplate="SMA rapide: $%{y:.2f}<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA_slow"],
        name=f"SMA {df.attrs.get('long_w', 'lente')}",
        line=dict(color="#7F77DD", width=1.5),
        hovertemplate="SMA lente: $%{y:.2f}<extra></extra>"
    ), row=1, col=1)

    if buy_signals:
        bx, by = zip(*buy_signals)
        fig.add_trace(go.Scatter(
            x=list(bx), y=list(by), mode="markers", name="Achat",
            marker=dict(color="#1D9E75", size=9, symbol="triangle-up"),
            hovertemplate="%{x|%d %b %Y}<br>Achat: $%{y:.2f}<extra></extra>"
        ), row=1, col=1)

    if sell_signals:
        sx, sy = zip(*sell_signals)
        fig.add_trace(go.Scatter(
            x=list(sx), y=list(sy), mode="markers", name="Vente",
            marker=dict(color="#E24B4A", size=9, symbol="triangle-down"),
            hovertemplate="%{x|%d %b %Y}<br>Vente: $%{y:.2f}<extra></extra>"
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["portfolio"], name="Stratégie",
        line=dict(color="#1D9E75", width=2),
        hovertemplate="%{x|%d %b %Y}<br>Portefeuille: $%{y:,.0f}<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["bh"], name="Buy & Hold",
        line=dict(color="#B4B2A9", width=1.5, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>Buy & Hold: $%{y:,.0f}<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=metrics["drawdown"] * 100,
        name="Drawdown", fill="tozeroy",
        line=dict(color="#E24B4A", width=1),
        fillcolor="rgba(226,75,74,0.15)",
        hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.1f}%<extra></extra>"
    ), row=3, col=1)

    fig.update_layout(
        height=760, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=70, b=30, l=0, r=0),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    fig.update_yaxes(tickprefix="$", row=1, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=2, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(ticksuffix="%", row=3, col=1, gridcolor="#f0f0f0")
    fig.update_xaxes(gridcolor="#f0f0f0")

    return fig


def plot_trade_returns(trades: list):
    """Bar chart des rendements de chaque trade, coloré vert/rouge."""
    if not trades:
        return None

    df_t = pd.DataFrame(trades)
    colors = ["#1D9E75" if r > 0 else "#E24B4A" for r in df_t["Rendement (%)"]]
    labels = [f"T{i+1}" for i in range(len(df_t))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=df_t["Rendement (%)"],
        marker_color=colors,
        text=[f"{r:+.1f}%" for r in df_t["Rendement (%)"]],
        textposition="outside",
        customdata=df_t[["Entrée", "Sortie", "Raison"]].values,
        hovertemplate=(
            "<b>Trade %{x}</b><br>"
            "Entrée : %{customdata[0]|%d %b %Y}<br>"
            "Sortie : %{customdata[1]|%d %b %Y}<br>"
            "Rendement : %{y:+.2f}%<br>"
            "Raison : %{customdata[2]}<extra></extra>"
        )
    ))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.update_layout(
        height=320,
        margin=dict(t=30, b=30, l=0, r=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Trades",
        yaxis_title="Rendement (%)",
        yaxis=dict(ticksuffix="%", gridcolor="#f0f0f0"),
        xaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig


def plot_walk_forward(wf_df: pd.DataFrame):
    """Graphique comparatif stratégie vs B&H par fenêtre + Sharpe."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.12,
        subplot_titles=(
            "Rendement stratégie vs Buy & Hold par fenêtre",
            "Ratio de Sharpe par fenêtre"
        )
    )

    windows = wf_df["Fenêtre"]

    fig.add_trace(go.Bar(
        x=windows, y=wf_df["Rdt strategie (%)"],
        name="Stratégie",
        marker=dict(color=["#1D9E75" if v >= 0 else "#E24B4A" for v in wf_df["Rdt strategie (%)"]]),
        text=[f"{v:+.1f}%" for v in wf_df["Rdt strategie (%)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Stratégie: %{y:+.2f}%<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=windows, y=wf_df["Rdt B&H (%)"],
        name="Buy & Hold",
        marker=dict(color="rgba(180,178,169,0.7)"),
        text=[f"{v:+.1f}%" for v in wf_df["Rdt B&H (%)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Buy & Hold: %{y:+.2f}%<extra></extra>"
    ), row=1, col=1)

    sharpe_colors = ["#378ADD" if v >= 0 else "#E24B4A" for v in wf_df["Sharpe"]]
    fig.add_trace(go.Bar(
        x=windows, y=wf_df["Sharpe"],
        name="Sharpe",
        marker=dict(color=sharpe_colors),
        text=[f"{v:.2f}" for v in wf_df["Sharpe"]],
        textposition="outside",
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>"
    ), row=2, col=1)

    fig.add_hline(y=0, line_color="#888", line_width=1, row=1, col=1)
    fig.add_hline(y=1, line_color="#EF9F27", line_width=1.5, line_dash="dot",
                  annotation_text="Sharpe = 1", annotation_position="right", row=2, col=1)
    fig.add_hline(y=0, line_color="#888", line_width=1, row=2, col=1)

    fig.update_layout(
        height=520, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=60, b=30, l=0, r=0),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    fig.update_yaxes(ticksuffix="%", gridcolor="#f0f0f0", row=1, col=1)
    fig.update_yaxes(gridcolor="#f0f0f0", row=2, col=1)
    fig.update_xaxes(gridcolor="#f0f0f0")

    return fig


def plot_optimization_heatmap(grid_df: pd.DataFrame, metric: str, metric_label: str,
                              current=None, best=None):
    """Heatmap de la métrique choisie en fonction des fenêtres SMA.

    current : tuple (fast, slow) des paramètres actuels de l'utilisateur
    best    : tuple (fast, slow) de la meilleure combinaison trouvée
    """
    pivot = grid_df.pivot(index="slow", columns="fast", values=metric)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        z=pivot.values,
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title=metric_label),
        hovertemplate=(
            "SMA rapide: %{x}j<br>SMA lente: %{y}j<br>"
            + metric_label + ": %{z}<extra></extra>"
        ),
    ))

    if best is not None:
        fig.add_trace(go.Scatter(
            x=[best[0]], y=[best[1]], mode="markers",
            marker=dict(symbol="star", size=16, color="#FFD700",
                        line=dict(color="#333333", width=1)),
            name="Meilleure combinaison",
            hovertemplate=f"Meilleur: SMA {best[0]}/{best[1]}<extra></extra>"
        ))

    if current is not None:
        fig.add_trace(go.Scatter(
            x=[current[0]], y=[current[1]], mode="markers",
            marker=dict(symbol="x", size=13, color="#1a1a1a"),
            name="Paramètres actuels",
            hovertemplate=f"Actuel: SMA {current[0]}/{current[1]}<extra></extra>"
        ))

    fig.update_layout(
        height=520,
        margin=dict(t=30, b=30, l=0, r=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="SMA rapide (jours)",
        yaxis_title="SMA lente (jours)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def plot_return_distribution(metrics: dict):
    """Histogramme des rendements journaliers avec les lignes de VaR."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=metrics["daily_returns"] * 100,
        nbinsx=60, name="Rendements",
        marker=dict(color="#378ADD"), opacity=0.7
    ))
    fig.add_vline(
        x=metrics["var_historical"], line_dash="dash",
        line_color="#E24B4A", line_width=2,
        annotation_text=f"VaR 95% = {metrics['var_historical']:.2f}%",
        annotation_position="top right"
    )
    fig.add_vline(
        x=metrics["var_parametric"], line_dash="dot",
        line_color="#7F77DD", line_width=2,
        annotation_text=f"VaR paramétrique = {metrics['var_parametric']:.2f}%",
        annotation_position="top left"
    )
    fig.update_layout(
        height=360, margin=dict(t=50, b=30, l=0, r=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Rendement journalier (%)",
        yaxis_title="Fréquence", showlegend=False
    )
    fig.update_xaxes(gridcolor="#f0f0f0")
    fig.update_yaxes(gridcolor="#f0f0f0")
    return fig


def plot_monte_carlo(sims, p5, p25, p50, p75, p95):
    """Trajectoires Monte Carlo avec bandes de confiance et percentiles."""
    n_days, n_sims = sims.shape
    future_days = list(range(n_days))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_days + future_days[::-1],
        y=list(p95) + list(p5[::-1]),
        fill="toself", fillcolor="rgba(55,138,221,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalle 90%", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=future_days + future_days[::-1],
        y=list(p75) + list(p25[::-1]),
        fill="toself", fillcolor="rgba(55,138,221,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalle 50%", hoverinfo="skip"
    ))
    for i in range(min(50, n_sims)):
        fig.add_trace(go.Scatter(
            x=future_days, y=sims[:, i],
            line=dict(color="rgba(55,138,221,0.08)", width=1),
            showlegend=False, hoverinfo="skip"
        ))
    fig.add_trace(go.Scatter(
        x=future_days, y=p50,
        line=dict(color="#378ADD", width=2), name="Médiane",
        hovertemplate="Jour %{x}<br>Prix médian: $%{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=future_days, y=p95,
        line=dict(color="#1D9E75", width=1.5, dash="dash"), name="95e percentile",
        hovertemplate="Jour %{x}<br>P95: $%{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=future_days, y=p5,
        line=dict(color="#E24B4A", width=1.5, dash="dash"), name="5e percentile",
        hovertemplate="Jour %{x}<br>P5: $%{y:.2f}<extra></extra>"
    ))
    fig.update_layout(
        height=400, hovermode="x unified",
        margin=dict(t=20, b=20, l=0, r=0),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Prix ($)", xaxis_title="Jours dans le futur",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0)
    )
    fig.update_xaxes(gridcolor="#f0f0f0")
    fig.update_yaxes(tickprefix="$", gridcolor="#f0f0f0")
    return fig
