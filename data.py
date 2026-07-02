"""Chargement des données de marché via Yahoo Finance.

Module volontairement sans dépendance à Streamlit : il peut être réutilisé
dans un notebook, un script ou des tests unitaires.
"""
import pandas as pd
import yfinance as yf


def load_data(ticker: str, start, end):
    """Télécharge les prix de clôture ajustés d'un ticker.

    Retourne (df, None) en cas de succès, (None, message_erreur) sinon.
    """
    try:
        if not ticker:
            return None, "Le ticker est vide."

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        if start_ts >= end_ts:
            return None, "La date de début doit être antérieure à la date de fin."

        end_ts = end_ts + pd.Timedelta(days=1)

        df = yf.download(
            ticker,
            start=start_ts,
            end=end_ts,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df is None or df.empty:
            return None, f"Aucune donnée retournée pour {ticker}."

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            return None, f"Colonne 'Close' absente pour {ticker}. Colonnes: {list(df.columns)}"

        df = df[["Close"]].copy()
        df.columns = ["Close"]
        df.dropna(inplace=True)

        if df.empty:
            return None, f"Les données de clôture de {ticker} sont vides après nettoyage."

        return df, None

    except Exception as e:
        return None, f"Erreur lors du chargement des données pour {ticker}: {e}"
