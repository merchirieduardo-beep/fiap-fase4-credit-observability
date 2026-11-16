"""Etapa 3 — Dashboard de observabilidade (HTML estático, autocontido, sem CDN).

Lê monitoring/metrics_history.jsonl + monitoring/alerts.jsonl e gera reports/dashboard.html.
Uso: python -m src.dashboard
"""
from __future__ import annotations

import html
import json
from datetime import datetime

from .config import (ALERTS_FILE, AUC_DROP_ALERT, DISPARATE_IMPACT_MIN, METRICS_FILE, PSI_DRIFT,
                     PSI_WARN, REPORTS_DIR)
from .model import MODEL_CARD_PATH
from .observability import read_jsonl

OUT = REPORTS_DIR / "dashboard.html"

CSS = """
:root{--bg:#f3f0e8;--panel:#fbfaf6;--ink:#1b211f;--muted:#66706b;--line:#d9d4c7;--ok:#0f6e5c;
--warn:#b0730f;--bad:#b3261e;--ok-bg:#dcebe5;--warn-bg:#f5e6c4;--bad-bg:#f6d5d0;--s2:#3b5b8c}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#141917;--panel:#1c2320;--ink:#e8e6df;
--muted:#9aa39e;--line:#33403a;--ok:#55c2a8;--warn:#e3b04b;--bad:#f08379;--ok-bg:#17352e;
--warn-bg:#3d3115;--bad-bg:#44201c;--s2:#8fb0e6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0;letter-spacing:-.01em}h2{font-size:13px;text-transform:uppercase;
letter-spacing:.09em;color:var(--muted);margin:38px 0 12px;border-top:2px solid var(--ink);padding-top:10px}
.sub{color:var(--muted);margin:4px 0 0}.mono,td.n,th.n,.kpi b{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
font-variant-numeric:tabular-nums}
.verdict{margin:20px 0 0;padding:14px 18px;border-left:6px solid;border-radius:2px;font-weight:600}
.verdict.bad{background:var(--bad-bg);border-color:var(--bad)}.verdict.ok{background:var(--ok-bg);border-color:var(--ok)}
.verdict.warn{background:var(--warn-bg);border-color:var(--warn)}.verdict small{display:block;font-weight:400;margin-top:2px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line);margin-top:18px}
.kpi{background:var(--panel);padding:14px 16px}.kpi span{display:block;font-size:12px;color:var(--muted)}
.kpi b{font-size:26px;font-weight:600}.kpi i{font-style:normal;font-size:12px;color:var(--muted);display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);padding:14px 16px}
.card h3{margin:0 0 2px;font-size:15px}.card p{margin:0 0 8px;color:var(--muted);font-size:13px}
svg{width:100%;height:auto;display:block}svg text{fill:var(--muted);font-size:11px;font-family:ui-monospace,Menlo,monospace}
.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:7px 10px;border-bottom:1px solid var(--line);
text-align:left;white-space:nowrap}th{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
td.n,th.n{text-align:right}td.wrap{white-space:normal;min-width:320px}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.ok{background:var(--ok-bg);color:var(--ok)}.warn{background:var(--warn-bg);color:var(--warn)}
.bad{background:var(--bad-bg);color:var(--bad)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
a{color:var(--ok)}dl{display:grid;grid-template-columns:minmax(180px,260px) 1fr;gap:6px 18px;font-size:13.5px;margin:0}
dt{font-weight:600}dd{margin:0;color:var(--muted)}@media(max-width:640px){dl{grid-template-columns:1fr}}
"""


def esc(value) -> str:
    return html.escape(str(value))


def pct(v, d=1) -> str:
    return "—" if v is None else f"{v * 100:.{d}f}%"


def tag(text: str, kind: str) -> str:
    return f'<span class="tag {kind}">{esc(text)}</span>'


def line_chart(labels: list[str], series: list[dict], refs: list[dict] | None = None,
               fmt=lambda v: f"{v:.2f}") -> str:
    """Gráfico de linhas em SVG puro. series: {name, values, color, dash?}; refs: {value, label, color}."""
    w, h, ml, mr, mt, mb = 520, 230, 46, 30, 12, 28
    vals = [v for s in series for v in s["values"] if v is not None] + [r["value"] for r in refs or []]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.15 or 0.05
    lo, hi = (max(lo - pad, 0) if lo >= 0 else lo - pad), hi + pad
    x = lambda i: ml + (w - ml - mr) * (i / max(len(labels) - 1, 1))  # noqa: E731
    y = lambda v: mt + (h - mt - mb) * (1 - (v - lo) / (hi - lo))      # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--line)"/>'
                   f'<text x="{ml - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{fmt(v)}</text>')
    for i, lab in enumerate(labels):
        out.append(f'<text x="{x(i):.1f}" y="{h - 8}" text-anchor="middle">{esc(lab)}</text>')
    for r in refs or []:
        out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y(r["value"]):.1f}" y2="{y(r["value"]):.1f}" '
                   f'stroke="{r["color"]}" stroke-dasharray="5 4"/>'
                   f'<text x="{w - mr - 4}" y="{y(r["value"]) - 5:.1f}" text-anchor="end" '
                   f'style="fill:{r["color"]}">{esc(r["label"])}</text>')
    for s in series:
        pts = [(x(i), y(v)) for i, v in enumerate(s["values"]) if v is not None]
        path = " ".join(f"{'M' if k == 0 else 'L'}{px:.1f},{py:.1f}" for k, (px, py) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        out.append(f'<path d="{path}" fill="none" stroke="{s["color"]}" stroke-width="2.4"{dash}/>')
        out += [f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" fill="{s["color"]}"/>' for px, py in pts]
    out.append("</svg>")
    legend = "".join(f'<span><i style="background:{s["color"]}"></i>{esc(s["name"])}</span>' for s in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


def psi_cell(v: float) -> str:
    kind = "bad" if v >= PSI_DRIFT else "warn" if v >= PSI_WARN else ""
    return f'<td class="n {kind}">{v:.3f}</td>'


def main():
    history = read_jsonl(METRICS_FILE)
    alerts = read_jsonl(ALERTS_FILE)
    card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
    base = card["baseline_metrics"]
    ok = [r for r in history if r["contract_status"] == "aprovado"]
    blocked = [r for r in history if r["contract_status"] != "aprovado"]
    last = ok[-1]
    labels = [r["batch"] for r in ok]
    perf = lambda key: [(r.get("performance") or {}).get(key) for r in ok]  # noqa: E731
    critical = [a for a in alerts if a["severity"] == "critico"]

    # ---- veredito
    last_alerts = {a["type"] for a in alerts if a["batch"] == last["batch"]}
    if last_alerts & {"retrain_recommended", "performance_degradation", "concept_drift"}:
        verdict = ("bad", f"MODELO DEGRADADO no lote {last['batch']} — retreino recomendado",
                   f"{last['n_drifted_features']} de {len(last['psi'])} features com drift · "
                   f"AUC {base['auc']:.3f} → {last['performance']['auc']:.3f} · inadimplência entre aprovados "
                   f"{pct(base['bad_rate_approved'])} → {pct(last['performance']['bad_rate_approved'])}")
    elif last["n_alerts"]:
        verdict = ("warn", f"Atenção no lote {last['batch']}", f"{last['n_alerts']} alerta(s) emitido(s)")
    else:
        verdict = ("ok", "Modelo estável", "Nenhum alerta no último lote")

    kpis = [
        ("Lotes recebidos", len(history), f"{len(blocked)} bloqueado(s) pelo contrato"),
        ("Linhas ingeridas", f"{sum(r['rows_ingested'] for r in history):,}".replace(",", "."),
         f"de {sum(r['rows_received'] for r in history):,} recebidas".replace(",", ".")),
        ("Alertas", len(alerts), f"{len(critical)} críticos"),
        ("Features com drift", f"{last['n_drifted_features']}/{len(last['psi'])}", f"lote {last['batch']}"),
        ("AUC atual", f"{last['performance']['auc']:.3f}", f"baseline {base['auc']:.3f}"),
        ("Latência de scoring", f"{last['latency_per_1k_rows_ms']:.0f} ms", "por 1.000 linhas"),
    ]
    kpi_html = "".join(f'<div class="kpi"><span>{esc(a)}</span><b>{esc(b)}</b><i>{esc(c)}</i></div>'
                       for a, b, c in kpis)

    # ---- gráficos
    charts = [
        ("Poder discriminante (AUC)", "Calculado quando os rótulos reais amadurecem.",
         line_chart(labels, [{"name": "AUC em produção", "values": perf("auc"), "color": "var(--ok)"}],
                    [{"value": base["auc"], "label": "baseline", "color": "var(--muted)"},
                     {"value": base["auc"] - AUC_DROP_ALERT, "label": "limite de alerta", "color": "var(--bad)"}],
                    lambda v: f"{v:.3f}")),
        ("Risco real × risco previsto", "A distância entre as linhas é o quanto o modelo subestima o risco (concept drift).",
         line_chart(labels, [{"name": "Inadimplência real", "values": perf("default_rate"), "color": "var(--bad)"},
                             {"name": "Score médio previsto", "values": perf("mean_score"), "color": "var(--s2)", "dash": True}],
                    fmt=lambda v: f"{v * 100:.0f}%")),
        ("Impacto no negócio", "Inadimplência entre clientes APROVADOS pelo modelo.",
         line_chart(labels, [{"name": "Inadimplência entre aprovados", "values": perf("bad_rate_approved"), "color": "var(--bad)"}],
                    [{"value": base["bad_rate_approved"], "label": "baseline", "color": "var(--muted)"}],
                    lambda v: f"{v * 100:.0f}%")),
        ("Features com drift e alertas", "Contagem por lote (PSI ≥ %.2f)." % PSI_DRIFT,
         line_chart(labels, [{"name": "Features com drift", "values": [r["n_drifted_features"] for r in ok], "color": "var(--warn)"},
                             {"name": "Alertas", "values": [r["n_alerts"] for r in ok], "color": "var(--bad)", "dash": True}],
                    fmt=lambda v: f"{v:.0f}")),
    ]
    chart_html = "".join(f'<div class="card"><h3>{esc(t)}</h3><p>{esc(d)}</p>{svg}</div>' for t, d, svg in charts)

    # ---- heatmap de PSI
    feats = sorted(last["psi"], key=lambda f: -last["psi"][f])
    psi_rows = "".join(
        f"<tr><td class='mono'>{esc(f)}</td>" + "".join(psi_cell(r["psi"][f]) for r in ok) +
        f"<td class='n'>{last['ks_pvalue'][f]:.2g}</td></tr>" for f in feats)
    psi_rows += ("<tr><td class='mono'><b>score (predição)</b></td>" +
                 "".join(psi_cell(r["score_psi"]) for r in ok) + "<td class='n'>—</td></tr>")
    psi_head = "".join(f"<th class='n'>{esc(b)}</th>" for b in labels)

    # ---- saúde do pipeline
    def run_row(r):
        good = r["contract_status"] == "aprovado"
        return (f"<tr><td class='mono'>{esc(r['batch'])}</td>"
                f"<td>{tag('sucesso', 'ok') if r['status'] == 'sucesso' else tag('falha', 'bad')}</td>"
                f"<td>{tag('aprovado', 'ok') if good else tag(r['contract_status'], 'bad')}</td>"
                f"<td class='n'>{r['rows_received']}</td><td class='n'>{r['rows_ingested']}</td>"
                f"<td class='n'>{r['contract_failures']}</td>"
                f"<td class='n'>{r['latency_validation_ms']:.0f}</td>"
                f"<td class='n'>{r.get('latency_scoring_ms', 0):.0f}</td>"
                f"<td class='n'>{r.get('latency_total_ms', 0):.0f}</td>"
                f"<td class='n'>{r['n_alerts']}</td></tr>")
    runs = "".join(run_row(r) for r in history)

    alert_rows = "".join(
        f"<tr><td class='mono'>{esc(a['batch'])}</td><td>{tag(a['severity'], 'bad' if a['severity'] == 'critico' else 'warn')}</td>"
        f"<td class='mono'>{esc(a['type'])}</td><td class='wrap'>{esc(a['message'])}</td></tr>"
        for a in reversed(alerts)) or "<tr><td colspan=4>Nenhum alerta.</td></tr>"

    # ---- viés
    def di_cell(v):
        return f'<td class="n {"warn" if v < DISPARATE_IMPACT_MIN else ""}">{v:.2f}</td>'
    bf = card.get("baseline_fairness", {})
    fair_rows = "".join(
        f"<tr><td>{esc(name)}</td>{di_cell(bf[g]['disparate_impact']) if g in bf else '<td></td>'}"
        + "".join(di_cell(r["fairness"][g]["disparate_impact"]) for r in ok) + "</tr>"
        for g, name in (("gender", "Gênero"), ("age_band", "Faixa etária")))

    # ---- análise causal
    causal_html = ""
    causal_path = REPORTS_DIR / "causal_analysis.json"
    if causal_path.exists():
        causal = json.loads(causal_path.read_text(encoding="utf-8"))
        sc = "".join(f"<tr><td>{esc(n)}</td><td class='n'>{m['auc']:.3f}</td><td class='n'>{pct(m['default_rate'])}</td>"
                     f"<td class='n'>{pct(m['mean_score'])}</td><td class='n'>{pct(m['bad_rate_approved'])}</td></tr>"
                     for n, m in causal["scenarios"].items())
        ab = "".join(f"<tr><td>{esc(r['mecanismo'])}</td><td class='n'>{r['delta_auc']:+.3f}</td>"
                     f"<td class='n'>{r['delta_bad_rate_approved'] * 100:+.1f} pp</td>"
                     f"<td class='n'>{r['risco_subestimado'] * 100:+.1f} pp</td></tr>" for r in causal["ablation"])
        causal_html = f"""<h2>Análise causal — por que degradou?</h2>
<div class="scroll"><table><tr><th>Cenário (intervenção)</th><th class="n">AUC</th><th class="n">Inad. real</th>
<th class="n">Previsto</th><th class="n">Inad. aprovados</th></tr>{sc}</table></div>
<div class="scroll" style="margin-top:16px"><table><tr><th>Mecanismo isolado</th><th class="n">ΔAUC</th><th class="n">Δ inad. aprovados</th>
<th class="n">Risco subestimado</th></tr>{ab}</table></div>"""

    # ---- links para evidências
    links = []
    for folder, title in (("evidently", "Evidently AI"), ("contracts", "Contrato de dados")):
        for f in sorted((REPORTS_DIR / folder).glob("*.html")):
            links.append(f'<li><a href="{folder}/{esc(f.name)}">{title} — {esc(f.name)}</a></li>')
    links_html = "".join(links) or "<li>Nenhum relatório encontrado.</li>"

    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Observabilidade — Credit Scoring</title><style>{CSS}</style></head><body><main>
<h1>Observabilidade do modelo de Credit Scoring</h1>
<p class="sub">Modelo <span class="mono">{esc(card['model_name'])}</span> · {esc(card['algorithm'])} ·
referência de {esc(card['reference_snapshot'])} · contrato v{esc(card['data_contract_version'])} ·
gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
<div class="verdict {verdict[0]}">{esc(verdict[1])}<small>{esc(verdict[2])}</small></div>
<div class="kpis">{kpi_html}</div>

<h2>Saúde do modelo ao longo do tempo</h2><div class="grid">{chart_html}</div>

<h2>Drift por feature — PSI (Referência × lote de produção)</h2>
<div class="scroll"><table><tr><th>Feature</th>{psi_head}<th class="n">p-valor KS/χ² (último)</th></tr>{psi_rows}</table></div>
<p class="sub">PSI &lt; {PSI_WARN:.2f} estável · {PSI_WARN:.2f}–{PSI_DRIFT:.2f} {tag('atenção', 'warn')} · ≥ {PSI_DRIFT:.2f} {tag('drift', 'bad')}</p>

<h2>Saúde do pipeline de dados</h2>
<div class="scroll"><table><tr><th>Lote</th><th>Execução</th><th>Contrato</th><th class="n">Recebidas</th><th class="n">Ingeridas</th>
<th class="n">Violações</th><th class="n">Validação (ms)</th><th class="n">Scoring (ms)</th><th class="n">Total (ms)</th><th class="n">Alertas</th></tr>{runs}</table></div>

<h2>Alertas ({len(alerts)})</h2>
<div class="scroll"><table><tr><th>Lote</th><th>Severidade</th><th>Tipo</th><th>Mensagem</th></tr>{alert_rows}</table></div>
{causal_html}
<h2>Auditoria de viés — razão de impacto na aprovação (mín/máx entre grupos)</h2>
<div class="scroll"><table><tr><th>Atributo</th><th class="n">Baseline</th>{psi_head}</tr>{fair_rows}</table></div>
<p class="sub">Abaixo de {DISPARATE_IMPACT_MIN:.2f} viola a regra dos 4/5. Gênero não é feature do modelo (só auditoria). Idade é feature, por isso a faixa etária é auditada a cada lote.</p>

<h2>Relatórios detalhados</h2><ul>{links_html}</ul>

<h2>Dicionário de métricas</h2><dl>
<dt>Execução / Contrato</dt><dd>Status de cada tarefa do pipeline e do contrato de dados (Pandera). Lote reprovado é bloqueado por inteiro.</dd>
<dt>Recebidas / Ingeridas</dt><dd>Volume de dados que chegou e que de fato alcançou o modelo.</dd>
<dt>Latência</dt><dd>Tempo de validação, scoring e total do lote (ms).</dd>
<dt>PSI</dt><dd>Population Stability Index por feature contra a referência de treino.</dd>
<dt>KS / χ²</dt><dd>Teste de hipótese de igualdade de distribuições (numéricas / categóricas).</dd>
<dt>AUC, inadimplência entre aprovados</dt><dd>Qualidade do modelo com rótulos maduros e seu impacto na carteira.</dd>
<dt>Alertas</dt><dd>Quantidade de alertas de contrato, drift, performance e viés emitidos por lote.</dd></dl>
</main></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(main())
