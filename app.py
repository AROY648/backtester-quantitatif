"""SMA Momentum Backtester — interface Streamlit.

Ce fichier ne contient que l'interface et l'orchestration. La logique métier
vit dans les modules :
- data.py     : chargement des données (yfinance)
- backtest.py : moteur de backtest, walk-forward, Monte Carlo
- metrics.py  : métriques globales et par trade
- plots.py    : graphiques Plotly
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

import data
import plots
from backtest import run_backtest, walk_forward_analysis, monte_carlo, grid_search
from metrics import compute_metrics, compute_trade_metrics

st.set_page_config(page_title="SMA Momentum Backtester", layout="wide")

st.markdown("""
<style>
div[data-testid="metric-container"] {
    padding: 18px 20px;
    margin-bottom: 8px;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("SMA Momentum Backtester")
st.caption("Stratégie de trading par croisement de moyennes mobiles simples (golden cross / death cross)")


# Le cache Streamlit est appliqué ici (couche interface) plutôt que dans
# data.py, pour garder le module data réutilisable hors Streamlit.
@st.cache_data(ttl=3600)
def load_data(ticker: str, start, end):
    return data.load_data(ticker, start, end)


# Le préfixe _ sur _progress dit à st.cache_data d'ignorer ce paramètre pour
# le hachage : la barre de progression n'apparaît qu'au premier calcul, les
# reruns suivants sont servis instantanément depuis le cache.
@st.cache_data(ttl=3600, show_spinner=False)
def run_grid_search(df, fast_values, slow_values, capital, fees,
                    stop_loss, take_profit, _progress=None):
    return grid_search(df, fast_values, slow_values, capital, fees,
                       stop_loss, take_profit, progress_callback=_progress)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Paramètres")

    ticker = st.text_input("Ticker (ex: SPY, AAPL, MSFT)").upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Début", value=date(2018, 1, 1))
    with col2:
        end_date = st.date_input("Fin", value=date.today())

    sma_short = st.slider("SMA rapide (jours)", min_value=5, max_value=100, value=20, step=1)
    sma_long = st.slider("SMA lente (jours)", min_value=20, max_value=300, value=50, step=5)

    if sma_short >= sma_long:
        st.warning("La SMA rapide doit être plus courte que la SMA lente.")
        st.stop()

    capital = st.number_input(
        "Capital initial ($)",
        min_value=1000,
        max_value=1_000_000,
        value=10_000,
        step=1000
    )

    fees = st.slider(
        "Frais de transaction (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05
    ) / 100

    st.divider()
    st.subheader("🛡️ Gestion du risque")

    use_stop_loss = st.checkbox("Activer Stop-Loss", value=False)
    stop_loss_pct = None
    if use_stop_loss:
        sl_val = st.slider("Stop-Loss (%)", min_value=-30, max_value=-1, value=-10, step=1)
        stop_loss_pct = sl_val / 100
        st.caption(f"Position fermée si prix baisse de **{abs(sl_val)}%** depuis l'entrée.")

    use_take_profit = st.checkbox("Activer Take-Profit", value=False)
    take_profit_pct = None
    if use_take_profit:
        tp_val = st.slider("Take-Profit (%)", min_value=1, max_value=100, value=20, step=1)
        take_profit_pct = tp_val / 100
        st.caption(f"Position fermée si prix monte de **{tp_val}%** depuis l'entrée.")

    run = st.button("Lancer le backtest", type="primary", width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

# Le bouton "run" n'est True que lors du clic. Sans session_state, bouger un
# slider situé dans les résultats (fenêtres walk-forward, simulations Monte
# Carlo) relancerait le script avec run=False et ferait disparaître la page.
if run:
    st.session_state["backtest_on"] = True

if st.session_state.get("backtest_on"):
    with st.spinner(f"Chargement des données pour {ticker}..."):
        df, load_error = load_data(ticker, start_date, end_date)

    if load_error:
        st.error(load_error)
        st.stop()

    st.success(f"{len(df)} lignes chargées pour {ticker}.")
    st.caption(f"Période couverte : du {df.index.min().date()} au {df.index.max().date()}")

    min_required = sma_long + 10
    if len(df) < min_required:
        st.error(
            f"Pas assez de données pour {ticker}. "
            f"{len(df)} lignes chargées, il en faut au moins **{min_required}**."
        )
        st.stop()

    df, buy_signals, sell_signals, final_value, trades = run_backtest(
        df, sma_short, sma_long, capital, fees, stop_loss_pct, take_profit_pct
    )

    if df is None or df.empty:
        st.error("Le backtest n'a pas pu être exécuté.")
        st.stop()

    metrics = compute_metrics(df, capital)
    trade_metrics = compute_trade_metrics(trades)

    # ── Métriques globales ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric(
        "Rendement stratégie",
        f"{metrics['total_return']:+.1f}%",
        delta=f"{metrics['total_return'] - metrics['bh_return']:+.1f}% vs B&H"
    )
    col2.metric("Buy & Hold", f"{metrics['bh_return']:+.1f}%")
    col3.metric("Ratio de Sharpe", f"{metrics['sharpe']:.2f}")
    col4.metric("Max drawdown", f"{metrics['max_dd']:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4, gap="large")
    col5.metric("CAGR (annualisé)", f"{metrics['cagr']:+.1f}%")
    col6.metric("Ratio de Sortino", f"{metrics['sortino']:.2f}")
    col7.metric("Ratio de Calmar", f"{metrics['calmar']:.2f}")
    col8.metric("Frais payés", f"${df.attrs.get('total_fees', 0):,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col9, col10, col11 = st.columns(3, gap="large")
    col9.metric("VaR 95% (1j)", f"{metrics['var_historical']:.2f}%")
    col10.metric("VaR en $", f"${abs(metrics['var_dollar']):,.0f}")
    col11.metric("Nb de signaux", metrics["n_trades"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphique principal ───────────────────────────────────────────────────
    fig = plots.plot_results(df, buy_signals, sell_signals, ticker, metrics)
    st.plotly_chart(fig, width="stretch")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1 : ANALYSE TRADE PAR TRADE
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Analyse trade par trade")

    if not trades:
        st.info("Aucun trade exécuté sur cette période avec ces paramètres.")
    else:
        # ── Métriques résumées par trade ──────────────────────────────────────
        if trade_metrics:
            tc1, tc2, tc3, tc4, tc5, tc6 = st.columns(6, gap="small")
            tc1.metric("Trades complétés", trade_metrics["nb_trades"])
            tc2.metric("Win rate", f"{trade_metrics['win_rate']:.1f}%")
            tc3.metric("Gain moyen", f"{trade_metrics['avg_win']:+.2f}%")
            tc4.metric("Perte moyenne", f"{trade_metrics['avg_loss']:+.2f}%")
            tc5.metric("Profit factor",
                       f"{trade_metrics['profit_factor']:.2f}" if trade_metrics['profit_factor'] != float('inf') else "∞")
            tc6.metric("Durée moy.", f"{trade_metrics['avg_duration']:.0f}j")

            st.markdown("<br>", unsafe_allow_html=True)

            bc1, bc2 = st.columns(2, gap="large")
            bc1.metric("Meilleur trade", f"{trade_metrics['best_trade']:+.2f}%")
            bc2.metric("Pire trade", f"{trade_metrics['worst_trade']:+.2f}%")

        # ── Bar chart des rendements ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        fig_trades = plots.plot_trade_returns(trades)
        if fig_trades:
            st.plotly_chart(fig_trades, width="stretch")

        # ── Tableau détaillé ──────────────────────────────────────────────────
        with st.expander("📊 Voir le détail de tous les trades"):
            df_trades = pd.DataFrame(trades).copy()
            df_trades["Entrée"] = pd.to_datetime(df_trades["Entrée"]).dt.strftime("%d/%m/%Y")
            df_trades["Sortie"] = pd.to_datetime(df_trades["Sortie"]).dt.strftime("%d/%m/%Y")

            def color_return(val):
                if isinstance(val, (int, float)):
                    color = "#1D9E75" if val > 0 else ("#E24B4A" if val < 0 else "#888")
                    return f"color: {color}; font-weight: bold"
                return ""

            st.dataframe(
                df_trades.style
                .map(color_return, subset=["Rendement (%)", "P&L ($)"])
                .format({
                    "Prix entrée ($)": "${:.2f}",
                    "Prix sortie ($)": "${:.2f}",
                    "Rendement (%)": "{:+.2f}%",
                    "P&L ($)": "${:+,.2f}",
                    "Durée (j)": "{:.0f}j",
                }),
                width="stretch",
                hide_index=True
            )

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2 : WALK-FORWARD ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔄 Walk-Forward Analysis")
    st.markdown("""
    > **Concept clé** : un backtest sur une seule période peut être trompeur — la stratégie a peut-être
    > bien fonctionné *par chance* sur ce marché précis. Le walk-forward découpe les données en plusieurs
    > fenêtres indépendantes et mesure la **cohérence des performances dans le temps**.
    > Si la stratégie est robuste, elle devrait être profitable (ou au moins cohérente) sur la majorité des fenêtres.
    """)

    n_windows = st.slider("Nombre de fenêtres temporelles", min_value=2, max_value=10, value=4, step=1)

    with st.spinner("Calcul du walk-forward..."):
        wf_df, wf_error = walk_forward_analysis(
            df, sma_short, sma_long, capital, fees,
            stop_loss_pct, take_profit_pct,
            n_windows=n_windows
        )

    if wf_error:
        st.warning(wf_error)
    else:
        # ── Graphique walk-forward ────────────────────────────────────────────
        fig_wf = plots.plot_walk_forward(wf_df)
        st.plotly_chart(fig_wf, width="stretch")

        # ── Indicateur de robustesse ──────────────────────────────────────────
        n_beating_bh = int(np.sum(wf_df["Rdt strategie (%)"] > wf_df["Rdt B&H (%)"]))
        n_positive = int(np.sum(wf_df["Rdt strategie (%)"] > 0))
        total_w = len(wf_df)

        rob1, rob2, rob3 = st.columns(3, gap="large")
        rob1.metric(
            "Fenêtres > B&H",
            f"{n_beating_bh}/{total_w}",
            delta="robuste" if n_beating_bh >= total_w * 0.6 else "fragile"
        )
        rob2.metric(
            "Fenêtres positives",
            f"{n_positive}/{total_w}",
            delta="OK" if n_positive >= total_w * 0.6 else "à surveiller"
        )
        rob3.metric(
            "Sharpe moyen",
            f"{wf_df['Sharpe'].mean():.2f}"
        )

        # ── Tableau récapitulatif ─────────────────────────────────────────────
        with st.expander("📊 Tableau détaillé des fenêtres"):
            st.dataframe(
                wf_df.style.format({
                    "Rdt strategie (%)": "{:+.2f}%",
                    "Rdt B&H (%)": "{:+.2f}%",
                    "Sharpe": "{:.2f}",
                    "Max DD (%)": "{:.2f}%",
                    "Win rate (%)": "{:.1f}%",
                }),
                width="stretch",
                hide_index=True
            )

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 : OPTIMISATION DES PARAMÈTRES (GRID SEARCH)
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🗺️ Optimisation des paramètres")
    st.markdown("""
    > **Concept clé** : la heatmap teste toutes les combinaisons de SMA et colore chaque case
    > selon la performance. Une stratégie robuste montre une **zone** de bons paramètres
    > (un plateau vert) ; un **pic isolé** entouré de rouge est presque toujours de
    > l'overfitting — du bruit statistique qu'il ne faut pas trader.
    """)

    opt_c1, opt_c2 = st.columns(2)
    with opt_c1:
        resolution = st.radio(
            "Résolution de la grille",
            ["Grossière (~100 backtests, rapide)", "Fine (~300 backtests, plus lent)"],
        )
    with opt_c2:
        metric_options = {
            "Ratio de Sharpe": ("sharpe", "Sharpe"),
            "Rendement total (%)": ("total_return", "Rendement (%)"),
            "Ratio de Calmar": ("calmar", "Calmar"),
        }
        metric_choice = st.selectbox("Métrique à optimiser", list(metric_options.keys()))
        metric_col, metric_label = metric_options[metric_choice]

    if st.button("Lancer l'optimisation"):
        st.session_state["opt_on"] = True

    if st.session_state.get("opt_on"):
        if resolution.startswith("Grossière"):
            fast_values = tuple(range(5, 61, 5))
            slow_values = tuple(range(20, 201, 20))
        else:
            fast_values = tuple(range(5, 61, 3))
            slow_values = tuple(range(20, 201, 10))

        raw_df = load_data(ticker, start_date, end_date)[0]

        progress = st.progress(0.0, text="Grid search en cours...")
        grid_df = run_grid_search(
            raw_df, fast_values, slow_values, capital, fees,
            stop_loss_pct, take_profit_pct,
            _progress=lambda p: progress.progress(p, text=f"Grid search... {int(p * 100)}%")
        )
        progress.empty()

        if grid_df is None or grid_df.empty:
            st.warning("Aucune combinaison n'a pu être testée (période trop courte ?).")
        else:
            best_row = grid_df.loc[grid_df[metric_col].idxmax()]
            best_fast, best_slow = int(best_row["fast"]), int(best_row["slow"])

            fig_opt = plots.plot_optimization_heatmap(
                grid_df, metric_col, metric_label,
                current=(sma_short, sma_long),
                best=(best_fast, best_slow),
            )
            st.plotly_chart(fig_opt, width="stretch")

            current_value = {
                "sharpe": metrics["sharpe"],
                "total_return": metrics["total_return"],
                "calmar": metrics["calmar"],
            }[metric_col]

            oc1, oc2, oc3 = st.columns(3, gap="large")
            oc1.metric("Meilleure combinaison", f"SMA {best_fast} / {best_slow}")
            oc2.metric(f"{metric_label} optimal", f"{best_row[metric_col]:.2f}")
            oc3.metric(
                f"{metric_label} actuel (SMA {sma_short}/{sma_long})",
                f"{current_value:.2f}",
                delta=f"{current_value - best_row[metric_col]:+.2f} vs optimal"
            )

            st.caption(
                "⚠️ La meilleure combinaison sur le passé n'est pas une garantie pour le futur. "
                "Vérifie qu'elle se trouve dans une zone stable de la carte, puis valide-la avec "
                "le walk-forward. Le grid search utilise les mêmes frais et stop-loss/take-profit "
                "que le backtest principal."
            )

    # ═════════════════════════════════════════════════════════════════════════
    # DISTRIBUTION DES RENDEMENTS + VaR
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Distribution des rendements journaliers")

    fig_var = plots.plot_return_distribution(metrics)
    st.plotly_chart(fig_var, width="stretch")

    # ── Données brutes ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Voir les données brutes"):
        st.dataframe(
            df[["Close", "SMA_fast", "SMA_slow", "portfolio", "bh"]]
            .tail(100)
            .style.format({
                "Close": "${:.2f}",
                "SMA_fast": "${:.2f}",
                "SMA_slow": "${:.2f}",
                "portfolio": "${:,.0f}",
                "bh": "${:,.0f}"
            }),
            width="stretch"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # MONTE CARLO
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Simulation Monte Carlo — projections sur 1 an")

    n_sims = st.slider("Nombre de simulations", min_value=100, max_value=1000, value=500, step=100)

    with st.spinner("Calcul des simulations..."):
        sims, p5, p25, p50, p75, p95, mu_mc, sigma_mc = monte_carlo(
            load_data(ticker, start_date, end_date)[0],
            n_simulations=n_sims, n_days=252
        )

    last_price = float(df["Close"].iloc[-1])

    fig_mc = plots.plot_monte_carlo(sims, p5, p25, p50, p75, p95)
    st.plotly_chart(fig_mc, width="stretch")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prix actuel", f"${last_price:.2f}")
    col2.metric("Prix médian dans 1 an", f"${p50[-1]:.2f}",
                delta=f"{(p50[-1] / last_price - 1) * 100:+.1f}%")
    col3.metric("Scénario pessimiste (P5)", f"${p5[-1]:.2f}",
                delta=f"{(p5[-1] / last_price - 1) * 100:+.1f}%")
    col4.metric("Scénario optimiste (P95)", f"${p95[-1]:.2f}",
                delta=f"{(p95[-1] / last_price - 1) * 100:+.1f}%")

    st.caption(
        f"Rendement journalier moyen : {mu_mc * 100:.3f}% — "
        f"Volatilité journalière : {sigma_mc * 100:.3f}% "
        f"({sigma_mc * np.sqrt(252) * 100:.1f}% annualisée)"
    )

else:
    st.info("Configure les paramètres dans la barre latérale et clique sur **Lancer le backtest**.")

    st.markdown("""
    ### Comment ça fonctionne

    **Stratégie momentum SMA croisée :**
    - **Achat (golden cross)** : quand la SMA rapide passe au-dessus de la SMA lente → signal haussier
    - **Vente (death cross)** : quand la SMA rapide repasse en dessous → signal baissier

    **Exécution réaliste :** le croisement est détecté sur la clôture du jour t,
    mais le trade est exécuté au jour **t+1** — sinon on utiliserait une information
    qu'on ne pouvait pas connaître au moment de trader (*look-ahead bias*, le piège n°1 des backtests).

    **Métriques calculées :**
    - **Rendement total** : performance de la stratégie vs Buy & Hold
    - **CAGR** : rendement annualisé composé — comparable entre périodes différentes
    - **Ratio de Sharpe** : rendement ajusté au risque (> 1 = bon, > 2 = excellent)
    - **Ratio de Sortino** : comme Sharpe, mais ne pénalise que la volatilité à la baisse
    - **Ratio de Calmar** : CAGR / max drawdown — le "coût en douleur" du rendement
    - **Max drawdown** : perte maximale depuis un sommet
    - **Win rate (par trade)** : % de trades profitables
    - **Profit factor** : ratio gains bruts / pertes brutes (> 1 = profitable)

    **Gestion du risque :**
    - **Stop-Loss** : ferme automatiquement la position si le prix baisse trop depuis l'entrée
    - **Take-Profit** : sécurise les gains si le prix monte suffisamment

    **Walk-Forward Analysis :**
    - Divise la période en N fenêtres indépendantes
    - Vérifie si la stratégie est **robuste dans le temps** ou juste chanceuse sur une période donnée

    **Optimisation des paramètres (grid search) :**
    - Teste toutes les combinaisons de SMA et affiche une **heatmap** de performance
    - Une **zone stable** de bons paramètres = robustesse ; un **pic isolé** = overfitting
    """)
