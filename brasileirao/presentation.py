"""Identidade visual do Risco de degola."""
from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st


ASSETS = Path(__file__).resolve().parent.parent / "assets"


def render_header() -> None:
    st.markdown(f"<style>{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    mascot = b64encode((ASSETS / "fantasminha.png").read_bytes()).decode("ascii")
    st.markdown(
        '<section class="risk-hero">'
        '<div class="risk-hero-copy"><span class="risk-eyebrow">BRASILEIRÃO · SÉRIE A</span>'
        '<h1>Risco de <span class="risk-accent">degola.</span></h1>'
        '<p>Quem escapa do fantasma da Série B?</p>'
        '<p class="risk-description">Acompanhe as chances de rebaixamento e explore '
        'os caminhos de cada clube até o fim do campeonato.</p>'
        '<span class="risk-source">DADOS · FOOTBALL-DATA.ORG</span></div>'
        f'<img src="data:image/png;base64,{mascot}" '
        'alt="Fantasminha de lençol branco com a letra B no peito" />'
        '</section>', unsafe_allow_html=True,
    )


def render_risk_cards(summary, count: int) -> None:
    cards = []
    for row in summary.head(count).itertuples(index=False):
        risk = float(row.prob_rebaixamento)
        cards.append(
            '<article class="risk-card">'
            f'<div class="risk-club">{escape(str(row.clube))}</div>'
            f'<div class="risk-percent">{risk:.1f}<span>%</span></div>'
            '<div class="risk-card-label">de chance de queda</div>'
            f'<div class="risk-track"><div style="width:{min(max(risk, 0), 100):.2f}%"></div></div>'
            f'<div class="risk-points">{row.pontos_mediana:.0f} pontos · mediana projetada</div>'
            '</article>'
        )
    st.markdown('<div class="risk-cards">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def style_chart(figure):
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ECF5F4", colorway=["#72E2C1", "#FF827A", "#E9C46A", "#79BCE8", "#C8A4E3", "#ECA8C5"],
        margin=dict(l=12, r=12, t=60, b=16),
    )
    figure.update_xaxes(gridcolor="#23414A", zerolinecolor="#23414A")
    figure.update_yaxes(gridcolor="#23414A", zerolinecolor="#23414A")
    return figure
