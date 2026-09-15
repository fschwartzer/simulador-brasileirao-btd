from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from brasileirao.api import FootballDataError, fetch_brasileirao_matches
from brasileirao.config import resolve_api_token
from brasileirao.data import split_at_matchday, standings_from_results, team_catalog
from brasileirao.model import fit_davidson, select_davidson_hyperparameters
from brasileirao.presentation import render_header, render_risk_cards, style_chart
from brasileirao.simulation import simulate_season


st.set_page_config(page_title="Risco de degola", page_icon="👻", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def load_api_data(token: str, season: int) -> pd.DataFrame:
    return fetch_brasileirao_matches(token, season)


@st.cache_data(show_spinner=False)
def tune_model_parameters(completed: pd.DataFrame, team_ids: tuple[str, ...]):
    return select_davidson_hyperparameters(completed, team_ids)


def configured_token() -> str:
    try:
        return resolve_api_token(st.secrets, os.environ)
    except (FileNotFoundError, AttributeError):
        return resolve_api_token({}, os.environ)


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


render_header()

with st.sidebar:
    st.header("Seu cenário")
    st.caption("Explore o campeonato, rodada a rodada.")
    season = st.number_input("Temporada", min_value=2014, max_value=2100, value=datetime.now().year, step=1)
    st.caption("Fonte exclusiva: football-data.org · Série A")
    if st.button("Atualizar dados", width="stretch"):
        load_api_data.clear()

token = configured_token()
if not token:
    st.error("Conexão não configurada. Defina API_TOKEN nos Secrets do Streamlit Community Cloud.")
    st.stop()
try:
    with st.spinner("Buscando os jogos do Brasileirão…"):
        matches = load_api_data(token, int(season))
    teams = team_catalog(matches)
except (FootballDataError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

fetched_at = matches.attrs.get("fetched_at")
if fetched_at:
    consultation = pd.Timestamp(fetched_at).tz_convert("America/Sao_Paulo").strftime("%d/%m/%Y às %H:%M")
    st.sidebar.caption(f"Consulta à API: {consultation} (Brasília). Cache de até 15 minutos.")

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
    with st.expander("Ajustes do modelo", expanded=False):
        relegated_slots = st.number_input(
            "Vagas de rebaixamento", min_value=1, max_value=max(1, len(teams) - 1), value=min(4, len(teams) - 1)
        )
        automatic_tuning = st.checkbox(
            "Selecionar parâmetros por backtest temporal",
            value=True,
            help="Treina somente em rodadas anteriores e escolhe a configuração com menor log loss.",
        )
        regularization = 5.0
        home_advantage_regularization = 5.0
        home_regularization = 200.0
        decay_half_life = None
        if not automatic_tuning:
            regularization = st.slider(
                "Regularização das forças", 0.0, 10.0, 5.0, 0.25,
                help="Estabiliza estimativas; valores altos aproximam as forças dos clubes.",
            )
            home_advantage_regularization = st.slider(
                "Regularização do mando médio", 0.0, 20.0, 5.0, 0.25,
                help="Controla separadamente o efeito médio de jogar em casa no campeonato.",
            )
            home_regularization = float(
                st.select_slider(
                    "Regularização do mando por clube",
                    options=[0, 1, 5, 10, 20, 50, 100, 200],
                    value=50,
                    help="Encolhe os desvios de mando dos clubes em direção ao mando médio.",
                )
            )
            decay_half_life = st.selectbox(
                "Decaimento temporal",
                options=[None, 4.0, 6.0, 8.0, 12.0, 20.0, 38.0],
                index=0,
                format_func=lambda value: (
                    "Sem decaimento" if value is None else f"Meia-vida de {value:g} rodadas"
                ),
                help="Após a meia-vida escolhida, o peso de um jogo cai pela metade.",
            )
        seed = st.number_input("Semente aleatória", min_value=0, max_value=2_147_483_647, value=1970)

observed, remaining = split_at_matchday(matches, cutoff)
if observed.empty:
    st.warning("Nenhum jogo encerrado até o corte. As forças começam iguais e a incerteza estrutural é máxima.")

backtest_result = None
observed_matchdays = observed["matchday"].dropna().astype(int).nunique()
if automatic_tuning and not observed.empty and observed_matchdays >= 12:
    with st.spinner("Selecionando decaimento e regularizações por backtest temporal…"):
        try:
            backtest_result = tune_model_parameters(
                observed,
                tuple(teams["team_id"].astype(str).tolist()),
            )
        except ValueError as exc:
            st.warning(f"Backtest indisponível ({exc}). Usando parâmetros conservadores padrão.")
    if backtest_result is not None:
        regularization = backtest_result.regularization
        home_advantage_regularization = backtest_result.home_advantage_regularization
        home_regularization = backtest_result.home_regularization
        decay_half_life = backtest_result.decay_half_life
        decay_description = (
            "sem decaimento" if decay_half_life is None else f"meia-vida {decay_half_life:g} rodadas"
        )
        st.sidebar.caption(
            f"Selecionados: forças {regularization:g}; mando médio {home_advantage_regularization:g}; "
            f"mando por clube {home_regularization:g}; {decay_description}."
        )
elif automatic_tuning and not observed.empty:
    st.warning(
        "Ainda há poucas rodadas para um backtest temporal estável. "
        "Foram usados os parâmetros conservadores padrão."
    )

with st.spinner("Ajustando o modelo e simulando a temporada…"):
    model = fit_davidson(
        observed,
        teams["team_id"].astype(str).tolist(),
        regularization=float(regularization),
        home_regularization=float(home_regularization),
        home_advantage_regularization=float(home_advantage_regularization),
        decay_half_life=None if decay_half_life is None else float(decay_half_life),
        reference_matchday=int(cutoff),
    )
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
metric_1.metric("Rodada de corte", f"{cutoff} / {max_matchday}")
metric_2.metric("Jogos observados", len(observed))
metric_3.metric("Jogos a simular", len(remaining))
metric_4.metric("Cenários simulados", f"{int(n_simulations):,}".replace(",", "."))

tab_risk, tab_charts, tab_table, tab_model, tab_data = st.tabs(
    ["Radar da degola", "Projeção de pontos", "Classificação", "Como calculamos", "Dados da API"]
)

with tab_risk:
    st.subheader("Na mira do fantasma")
    st.caption(f"Os {min(4, len(teams))} maiores riscos no cenário selecionado. Probabilidades estimadas pelo modelo.")
    render_risk_cards(simulation.summary, count=min(4, len(teams)))
    st.subheader("O risco de cada clube")
    st.caption("P05–P95: intervalo central de 90% das pontuações simuladas, condicionado ao modelo ajustado.")
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
        file_name=f"risco_de_degola_{int(season)}_rodada_{cutoff}.csv",
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
        st.plotly_chart(style_chart(histogram), width="stretch")
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
    st.plotly_chart(style_chart(boxplot), width="stretch")

with tab_table:
    current_table = standings_from_results(observed, teams)
    st.dataframe(
        current_table.drop(columns="team_id").rename(columns={"team": "Clube"}),
        width="stretch",
        hide_index=True,
    )

with tab_model:
    st.write("As chances de queda são calculadas pelo modelo Bradley–Terry–Davidson e por simulações Monte Carlo, a partir dos jogos da API.")
    model_metric_1, model_metric_2 = st.columns(2)
    model_metric_1.metric("Mando (multiplicador)", f"{np.exp(model.home_advantage):.2f}×")
    model_metric_2.metric("Parâmetro de empate ν", f"{model.draw_parameter:.2f}")
    st.subheader("Especificação")
    st.latex(r"P(H)=\frac{a}{a+b+\nu\sqrt{ab}},\quad P(E)=\frac{\nu\sqrt{ab}}{a+b+\nu\sqrt{ab}},\quad P(A)=\frac{b}{a+b+\nu\sqrt{ab}}")
    st.latex(r"a=\exp(\theta_H+h+\delta_H),\qquad b=\exp(\theta_A)")
    temporal_description = (
        "O backtest não aplicou decaimento temporal neste corte."
        if decay_half_life is None
        else f"O peso de um jogo cai pela metade a cada {decay_half_life:g} rodadas."
    )
    st.write(
        "As forças θ, o mando médio h, os desvios de mando por clube δ e o parâmetro de empate ν são "
        "estimados por máxima verossimilhança penalizada e ponderada no tempo. As somas de θ e δ são zero; "
        f"forças, mando médio e mando por clube usam regularizações independentes. {temporal_description}"
    )
    if backtest_result is not None:
        st.subheader("Seleção temporal dos parâmetros")
        st.write(
            f"A configuração foi escolhida em {len(backtest_result.validation_matchdays)} folds expansivos "
            f"(rodadas {backtest_result.validation_matchdays[0]}–{backtest_result.validation_matchdays[-1]}), "
            f"com {backtest_result.n_validation_matches} partidas de validação. Nenhum fold usa resultados "
            "da própria rodada prevista ou de rodadas posteriores."
        )
        backtest_table = backtest_result.scores.head(10).rename(
            columns={
                "regularization": "Reg. forças",
                "home_advantage_regularization": "Reg. mando médio",
                "home_regularization": "Reg. mando clube",
                "decay_half_life": "Meia-vida",
                "log_loss": "Log loss",
                "brier_score": "Brier",
                "log_loss_se": "EP log loss",
                "convergence_rate": "Convergência",
                "n_matches": "Jogos",
            }
        )
        backtest_table["Meia-vida"] = backtest_table["Meia-vida"].map(
            lambda value: "Sem decaimento" if pd.isna(value) else f"{value:g}"
        )
        st.dataframe(
            backtest_table.style.format(
                {
                    "Log loss": "{:.4f}",
                    "Brier": "{:.4f}",
                    "EP log loss": "{:.4f}",
                    "Convergência": "{:.0%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    st.dataframe(
        model.strength_table(teams).drop(columns="team_id").rename(
            columns={
                "team": "Clube",
                "forca_log": "Força log",
                "forca_relativa": "Força relativa",
                "desvio_mando_log": "Desvio de mando (log)",
                "multiplicador_mando_clube": "Mando total do clube",
            }
        ).style.format(
            {
                "Força log": "{:.3f}",
                "Força relativa": "{:.3f}",
                "Desvio de mando (log)": "{:+.3f}",
                "Mando total do clube": "{:.2f}×",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "A força é associativa, não causal. O decaimento captura mudança recente apenas pelos resultados; "
        "lesões, escalações, calendário de outras competições e trocas de técnico não entram no modelo."
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
    "Risco de degola · Dados: football-data.org. Projeções condicionadas ao modelo, sem garantia de resultado. "
    "Critérios simulados: pontos, vitórias, saldo de gols e gols pró. Empates residuais são sorteados, pois cartões e confronto direto não estão no endpoint gratuito."
)
