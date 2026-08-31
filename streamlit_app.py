from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from brasileirao.api import FootballDataError, fetch_brasileirao_matches, parse_uploaded_csv
from brasileirao.data import make_demo_matches, split_at_matchday, standings_from_results, team_catalog
from brasileirao.model import fit_davidson
from brasileirao.simulation import simulate_season


st.set_page_config(page_title="Risco de rebaixamento — Brasileirão", page_icon="⚽", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def load_api_data(token: str, season: int) -> pd.DataFrame:
    return fetch_brasileirao_matches(token, season)


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return make_demo_matches()


def configured_token() -> str:
    environment_value = os.getenv("FOOTBALL_DATA_TOKEN", "")
    try:
        return str(st.secrets.get("FOOTBALL_DATA_TOKEN", environment_value))
    except (FileNotFoundError, AttributeError):
        return environment_value


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.drop(columns="team_id").copy()
    output = output.rename(
        columns={
            "clube": "Clube",
            "prob_rebaixamento": "Risco de rebaixamento (%)",
            "pontos_mediana": "Pontos (mediana)",
            "pontos_p05": "Pontos P05",
            "pontos_p95": "Pontos P95",
            "posicao_mediana": "Posição (mediana)",
        }
    )
    return output


def risk_background(value: float) -> str:
    alpha = min(max(float(value) / 100.0, 0.0), 1.0) * 0.55
    return f"background-color: rgba(220, 53, 69, {alpha:.3f})"


st.title("Risco de rebaixamento no Brasileirão")
st.caption("Bradley–Terry–Davidson com mando de campo, placares condicionais e Monte Carlo")

with st.sidebar:
    st.header("Dados e simulação")
    source = st.radio("Fonte", ["football-data.org", "CSV", "Demonstração"], index=2)
    season = st.number_input("Temporada", min_value=2014, max_value=2100, value=2026, step=1)
    token = ""
    uploaded = None
    if source == "football-data.org":
        token = st.text_input(
            "Chave da API",
            value=configured_token(),
            type="password",
            help="Crie uma chave gratuita em football-data.org/client/register.",
        )
    elif source == "CSV":
        uploaded = st.file_uploader("Partidas em CSV", type=["csv"])

try:
    if source == "football-data.org":
        if not token:
            st.info("Informe a chave gratuita na barra lateral ou selecione Demonstração.")
            st.stop()
        with st.spinner("Consultando a football-data.org…"):
            matches = load_api_data(token, int(season))
        is_demo = False
    elif source == "CSV":
        if uploaded is None:
            st.info("Envie um CSV no contrato descrito no README.")
            st.stop()
        matches = parse_uploaded_csv(uploaded)
        is_demo = False
    else:
        matches = load_demo_data()
        is_demo = True
except (FootballDataError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

teams = team_catalog(matches)
all_matchdays = matches["matchday"].dropna().astype(int)
finished_matchdays = matches.loc[matches["status"].isin(["FINISHED", "AWARDED"]), "matchday"].dropna().astype(int)
last_finished = int(finished_matchdays.max()) if not finished_matchdays.empty else int(all_matchdays.min())

with st.sidebar:
    min_matchday, max_matchday = int(all_matchdays.min()), int(all_matchdays.max())
    if min_matchday == max_matchday:
        cutoff = min_matchday
        st.number_input("Rodada de corte", value=cutoff, disabled=True)
    else:
        cutoff = st.slider(
            "Rodada de corte",
            min_value=min_matchday,
            max_value=max_matchday,
            value=last_finished,
            help="Somente jogos encerrados até esta rodada são usados no ajuste.",
        )
    n_simulations = st.select_slider(
        "Simulações",
        options=[1_000, 2_500, 5_000, 10_000, 20_000],
        value=10_000,
    )
    relegated_slots = st.number_input(
        "Vagas de rebaixamento", min_value=1, max_value=max(1, len(teams) - 1), value=min(4, len(teams) - 1)
    )
    regularization = st.slider(
        "Regularização das forças", 0.0, 2.0, 0.25, 0.05,
        help="Estabiliza estimativas nas primeiras rodadas; valores altos aproximam as forças.",
    )
    seed = st.number_input("Semente aleatória", min_value=0, max_value=2_147_483_647, value=1970)

observed, remaining = split_at_matchday(matches, cutoff)
if is_demo:
    st.warning("Modo demonstrativo: calendário e placares são inteiramente artificiais; não interprete os números como projeção real.")
if observed.empty:
    st.warning("Nenhum jogo encerrado até o corte. As forças começam iguais e a incerteza estrutural é máxima.")

with st.spinner("Ajustando o modelo e simulando a temporada…"):
    model = fit_davidson(observed, teams["team_id"].astype(str).tolist(), float(regularization))
    simulation = simulate_season(
        observed=observed,
        remaining=remaining,
        teams=teams,
        model=model,
        n_simulations=int(n_simulations),
        relegated_slots=int(relegated_slots),
        seed=int(seed),
    )

if not model.converged:
    st.error("O otimizador não convergiu. Não use os resultados sem revisar os dados e a regularização.")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Jogos observados", len(observed))
metric_2.metric("Jogos simulados por cenário", len(remaining))
metric_3.metric("Mando (multiplicador)", f"{np.exp(model.home_advantage):.2f}×")
metric_4.metric("Parâmetro de empate ν", f"{model.draw_parameter:.2f}")

tab_risk, tab_charts, tab_table, tab_model, tab_data = st.tabs(
    ["Risco", "Distribuições", "Classificação no corte", "Modelo", "Dados"]
)

with tab_risk:
    formatted = format_summary(simulation.summary)
    st.dataframe(
        formatted.style.format(
            {
                "Risco de rebaixamento (%)": "{:.2f}",
                "Pontos (mediana)": "{:.0f}",
                "Pontos P05": "{:.0f}",
                "Pontos P95": "{:.0f}",
                "Posição (mediana)": "{:.0f}",
            }
        ).map(risk_background, subset=["Risco de rebaixamento (%)"]),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Baixar resumo CSV",
        formatted.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"risco_rebaixamento_{int(season)}_rodada_{cutoff}.csv",
        mime="text/csv",
    )

with tab_charts:
    distribution = simulation.distributions
    default_clubs = simulation.summary.head(min(6, len(teams)))["clube"].tolist()
    selected_clubs = st.multiselect(
        "Clubes no histograma",
        sorted(distribution["clube"].unique()),
        default=default_clubs,
    )
    if selected_clubs:
        histogram = px.histogram(
            distribution.loc[distribution["clube"].isin(selected_clubs)],
            x="pontos",
            color="clube",
            barmode="overlay",
            opacity=0.55,
            histnorm="probability",
            labels={"pontos": "Pontuação final", "clube": "Clube", "probability": "Proporção"},
            title="Distribuição simulada da pontuação final",
        )
        histogram.update_layout(legend_title_text="Clube")
        st.plotly_chart(histogram, width="stretch")
    else:
        st.info("Selecione ao menos um clube para o histograma.")

    club_order = simulation.summary.sort_values("pontos_mediana")["clube"].tolist()
    boxplot = px.box(
        distribution,
        x="pontos",
        y="clube",
        category_orders={"clube": club_order},
        points=False,
        labels={"pontos": "Pontuação final", "clube": "Clube"},
        title="Incerteza da pontuação final por clube",
    )
    boxplot.update_layout(height=max(550, 27 * len(teams)))
    st.plotly_chart(boxplot, width="stretch")

with tab_table:
    current_table = standings_from_results(observed, teams)
    st.dataframe(
        current_table.drop(columns="team_id").rename(columns={"team": "Clube"}),
        width="stretch",
        hide_index=True,
    )

with tab_model:
    st.subheader("Especificação")
    st.latex(r"P(H)=\frac{a}{a+b+\nu\sqrt{ab}},\quad P(E)=\frac{\nu\sqrt{ab}}{a+b+\nu\sqrt{ab}},\quad P(A)=\frac{b}{a+b+\nu\sqrt{ab}}")
    st.latex(r"a=\exp(\theta_H+h),\qquad b=\exp(\theta_A)")
    st.write(
        "As forças θ, o mando h e o parâmetro de empate ν são estimados por máxima verossimilhança "
        "penalizada. A soma das forças é zero. O placar é amostrado condicionalmente ao resultado, usando "
        "os placares observados e um prior discreto suavizador."
    )
    st.dataframe(
        model.strength_table(teams).drop(columns="team_id").rename(
            columns={"team": "Clube", "forca_log": "Força log", "forca_relativa": "Força relativa"}
        ).style.format({"Força log": "{:.3f}", "Força relativa": "{:.3f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "A força é associativa, não causal. Lesões, escalações, calendário, trocas de técnico e dependência temporal não entram neste MVP."
    )

with tab_data:
    st.write(f"{len(matches)} partidas no calendário; {len(observed)} observadas no corte; {len(remaining)} simuladas.")
    st.dataframe(matches, width="stretch", hide_index=True)
    st.download_button(
        "Baixar dados normalizados",
        matches.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"partidas_brasileirao_{int(season)}.csv",
        mime="text/csv",
    )

st.caption(
    "Critérios simulados: pontos, vitórias, saldo de gols e gols pró. Empates residuais são sorteados, pois cartões e confronto direto não estão no endpoint gratuito."
)
