"""Calcul des métriques de performance et de risque.

Deux niveaux d'analyse :
- compute_metrics       : métriques globales sur la courbe de portefeuille
- compute_trade_metrics : statistiques sur les trades individuels
"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_metrics(df, capital):
    """Métriques globales calculées sur la colonne 'portfolio' du DataFrame."""
    final_val = float(df["portfolio"].iloc[-1])
    total_return = (final_val - capital) / capital * 100
    bh_return = (float(df["bh"].iloc[-1]) - capital) / capital * 100

    daily_returns = df["portfolio"].pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    rolling_max = df["portfolio"].cummax()
    drawdown = (df["portfolio"] - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100 if not drawdown.empty else 0.0

    # CAGR : rendement annualisé composé — comparable entre périodes de durées différentes
    n_years = len(df) / 252
    if n_years > 0 and final_val > 0:
        cagr = ((final_val / capital) ** (1 / n_years) - 1) * 100
    else:
        cagr = 0.0

    # Sortino : comme le Sharpe, mais ne pénalise que la volatilité à la baisse
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Calmar : rendement annualisé / pire drawdown — mesure le "coût en douleur" du rendement
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0

    n_trades = int((df["position"] != 0).sum())
    win_days = int((daily_returns > 0).sum())
    total_days = int(len(daily_returns))
    win_rate = (win_days / total_days * 100) if total_days > 0 else 0.0

    confidence = 0.95
    var_historical = float(np.percentile(daily_returns, (1 - confidence) * 100))

    mu = daily_returns.mean()
    sigma = daily_returns.std()
    var_parametric = float(stats.norm.ppf(1 - confidence, mu, sigma))

    var_dollar = var_historical * final_val

    return {
        "final_val": final_val,
        "total_return": total_return,
        "bh_return": bh_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cagr": cagr,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "drawdown": drawdown,
        "var_historical": var_historical * 100,
        "var_parametric": var_parametric * 100,
        "var_dollar": var_dollar,
        "daily_returns": daily_returns,
    }


def compute_trade_metrics(trades: list) -> dict:
    """Calcule des statistiques sur la liste des trades complétés."""
    if not trades:
        return {}

    df_t = pd.DataFrame(trades)
    # Exclure les positions encore ouvertes pour les stats de win/loss
    closed = df_t[df_t["Raison"] != "Fin période (ouvert)"]

    if closed.empty:
        return {}

    rets = closed["Rendement (%)"]
    wins = closed[rets > 0]
    losses = closed[rets <= 0]

    win_rate = len(wins) / len(closed) * 100 if len(closed) > 0 else 0.0
    avg_win = wins["Rendement (%)"].mean() if not wins.empty else 0.0
    avg_loss = losses["Rendement (%)"].mean() if not losses.empty else 0.0
    best = rets.max()
    worst = rets.min()
    avg_duration = closed["Durée (j)"].mean()

    gross_profit = wins["P&L ($)"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["P&L ($)"].sum()) if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "nb_trades": len(closed),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best,
        "worst_trade": worst,
        "avg_duration": avg_duration,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }
