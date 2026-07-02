"""Moteur de backtesting : stratégie SMA croisée, walk-forward, Monte Carlo.

Aucune dépendance à Streamlit — le moteur est testable indépendamment
de l'interface.
"""
import numpy as np
import pandas as pd

from metrics import compute_metrics, compute_trade_metrics


def run_backtest(df, short_w, long_w, capital, fees=0.0, stop_loss=None, take_profit=None):
    """
    Exécute le backtest SMA croisé.

    Paramètres
    ----------
    stop_loss   : float négatif (ex: -0.10 = -10%) ou None
    take_profit : float positif (ex: 0.20 = +20%) ou None

    Retourne
    --------
    df enrichi, buy_signals, sell_signals, final_value, trades (liste de dicts)
    """
    df = df.copy()
    df["SMA_fast"] = df["Close"].rolling(short_w).mean()
    df["SMA_slow"] = df["Close"].rolling(long_w).mean()
    df.dropna(inplace=True)

    df["signal"] = np.where(df["SMA_fast"] > df["SMA_slow"], 1, 0)
    # shift(1) : le croisement est détecté sur la clôture du jour t, donc le
    # trade ne peut être exécuté qu'au jour t+1 (sinon look-ahead bias).
    df["position"] = df["signal"].diff().shift(1).fillna(0)

    cash = float(capital)
    shares = 0.0
    portfolio_values = []
    buy_signals = []
    sell_signals = []
    total_fees_paid = 0.0

    # Variables pour le tracking des trades
    trades = []
    entry_date = None
    entry_price = None
    entry_shares = None   # nombre d'actions détenues au moment de l'achat

    for idx, row in df.iterrows():
        price = float(row["Close"])
        exit_reason = None

        # ── Vérification Stop-Loss / Take-Profit ──────────────────────────────
        if shares > 0 and entry_price is not None:
            change = (price - entry_price) / entry_price
            if stop_loss is not None and change <= stop_loss:
                exit_reason = "Stop-Loss"
            elif take_profit is not None and change >= take_profit:
                exit_reason = "Take-Profit"

        if exit_reason:
            gross = shares * price
            fee = gross * fees
            cash = gross - fee
            total_fees_paid += fee
            sell_signals.append((idx, price))

            trades.append({
                "Entrée": entry_date,
                "Sortie": idx,
                "Prix entrée ($)": round(entry_price, 2),
                "Prix sortie ($)": round(price, 2),
                "Rendement (%)": round((price / entry_price - 1) * 100, 2),
                "P&L ($)": round(entry_shares * (price - entry_price), 2),
                "Durée (j)": (idx - entry_date).days,
                "Raison": exit_reason,
            })
            shares = 0.0
            entry_date = None
            entry_price = None
            entry_shares = None

        # ── Signaux SMA ────────────────────────────────────────────────────────
        if row["position"] == 1 and cash > 0:
            fee = cash * fees
            cash -= fee
            total_fees_paid += fee
            shares = cash / price
            cash = 0.0
            buy_signals.append((idx, price))
            entry_date = idx
            entry_price = price
            entry_shares = shares

        elif row["position"] == -1 and shares > 0 and exit_reason is None:
            gross = shares * price
            fee = gross * fees
            cash = gross - fee
            total_fees_paid += fee
            sell_signals.append((idx, price))

            if entry_date is not None:
                trades.append({
                    "Entrée": entry_date,
                    "Sortie": idx,
                    "Prix entrée ($)": round(entry_price, 2),
                    "Prix sortie ($)": round(price, 2),
                    "Rendement (%)": round((price / entry_price - 1) * 100, 2),
                    "P&L ($)": round(entry_shares * (price - entry_price), 2),
                    "Durée (j)": (idx - entry_date).days,
                    "Raison": "Signal",
                })
            shares = 0.0
            entry_date = None
            entry_price = None
            entry_shares = None

        portfolio_values.append(cash + shares * price)

    # ── Position encore ouverte à la fin de la période ────────────────────────
    if shares > 0 and entry_date is not None:
        last_price = float(df["Close"].iloc[-1])
        last_idx = df.index[-1]
        trades.append({
            "Entrée": entry_date,
            "Sortie": last_idx,
            "Prix entrée ($)": round(entry_price, 2),
            "Prix sortie ($)": round(last_price, 2),
            "Rendement (%)": round((last_price / entry_price - 1) * 100, 2),
            "P&L ($)": round(entry_shares * (last_price - entry_price), 2),
            "Durée (j)": (last_idx - entry_date).days,
            "Raison": "Fin période (ouvert)",
        })

    final_value = cash + shares * float(df["Close"].iloc[-1])
    df["portfolio"] = portfolio_values
    df["bh"] = capital * df["Close"] / float(df["Close"].iloc[0])
    df.attrs["short_w"] = short_w
    df.attrs["long_w"] = long_w
    df.attrs["total_fees"] = total_fees_paid

    return df, buy_signals, sell_signals, final_value, trades


def walk_forward_analysis(df, short_w, long_w, capital, fees, stop_loss, take_profit, n_windows=5):
    """
    Découpe les données en n_windows fenêtres temporelles égales et exécute
    le backtest sur chacune.

    Objectif pédagogique : vérifier si la stratégie est robuste dans le temps
    ou si elle ne fonctionne que sur certaines périodes (overfitting).

    Retourne un DataFrame avec les métriques par fenêtre.
    """
    if len(df) < (long_w + 10) * n_windows:
        return None, "Pas assez de données pour le nombre de fenêtres demandé."

    window_size = len(df) // n_windows
    results = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size if i < n_windows - 1 else len(df)
        window_df = df.iloc[start_idx:end_idx].copy()

        # Il faut assez de données pour calculer les SMA
        if len(window_df) < long_w + 10:
            continue

        try:
            res_df, _, _, _, trades_w = run_backtest(
                window_df, short_w, long_w, capital, fees, stop_loss, take_profit
            )
            m = compute_metrics(res_df, capital)
            tm = compute_trade_metrics(trades_w)

            results.append({
                "Fenêtre": f"#{i+1}",
                "Début": window_df.index[0].strftime("%d/%m/%Y"),
                "Fin": window_df.index[-1].strftime("%d/%m/%Y"),
                "Rdt strategie (%)": round(m["total_return"], 2),
                "Rdt B&H (%)": round(m["bh_return"], 2),
                "Sharpe": round(m["sharpe"], 2),
                "Max DD (%)": round(m["max_dd"], 2),
                "Nb trades": tm.get("nb_trades", 0),
                "Win rate (%)": round(tm.get("win_rate", 0), 1),
            })
        except Exception:
            continue

    if not results:
        return None, "Aucune fenêtre n'a pu être calculée."

    return pd.DataFrame(results), None


def grid_search(df, fast_values, slow_values, capital, fees=0.0,
                stop_loss=None, take_profit=None, progress_callback=None):
    """
    Backteste toutes les combinaisons (SMA rapide, SMA lente) valides.

    Objectif pédagogique : visualiser la sensibilité aux paramètres. Une
    stratégie robuste montre une ZONE de bons paramètres ; un pic isolé
    entouré de mauvais résultats est un signe d'overfitting.

    progress_callback : fonction optionnelle appelée avec l'avancement (0 à 1)
    — permet à l'interface d'afficher une barre de progression sans que le
    moteur dépende de Streamlit.

    Retourne un DataFrame avec une ligne par combinaison testée, ou None.
    """
    combos = [(f, s) for f in fast_values for s in slow_values if f < s]
    if not combos:
        return None

    results = []
    for i, (f, s) in enumerate(combos):
        if len(df) >= s + 10:
            try:
                res_df, _, _, _, _ = run_backtest(df, f, s, capital, fees, stop_loss, take_profit)
                m = compute_metrics(res_df, capital)
                results.append({
                    "fast": f,
                    "slow": s,
                    "sharpe": round(m["sharpe"], 2),
                    "total_return": round(m["total_return"], 2),
                    "calmar": round(m["calmar"], 2),
                    "max_dd": round(m["max_dd"], 2),
                })
            except Exception:
                pass
        if progress_callback:
            progress_callback((i + 1) / len(combos))

    if not results:
        return None
    return pd.DataFrame(results)


def monte_carlo(df, n_simulations=500, n_days=252):
    """Simulation Monte Carlo de trajectoires de prix (mouvement brownien géométrique)."""
    close = df["Close"]
    daily_returns = close.pct_change().dropna()

    mu = daily_returns.mean()
    sigma = daily_returns.std()
    last_price = float(close.iloc[-1])

    # Version vectorisée : on génère tous les chocs aléatoires d'un coup,
    # puis cumsum des log-rendements — ~100x plus rapide que la double boucle.
    z = np.random.standard_normal((n_days - 1, n_simulations))
    log_returns = (mu - 0.5 * sigma**2) + sigma * z
    log_paths = np.vstack([np.zeros((1, n_simulations)), np.cumsum(log_returns, axis=0)])
    simulations = last_price * np.exp(log_paths)

    p5  = np.percentile(simulations, 5,  axis=1)
    p25 = np.percentile(simulations, 25, axis=1)
    p50 = np.percentile(simulations, 50, axis=1)
    p75 = np.percentile(simulations, 75, axis=1)
    p95 = np.percentile(simulations, 95, axis=1)

    return simulations, p5, p25, p50, p75, p95, mu, sigma
