"""Rebuild the project's technical report from measured benchmark JSON.

Usage (from repository root):
    python -m pip install 'reportlab>=4.0,<5'
    python scripts/make_technical_report.py

The script intentionally never runs benchmarks or silently invents measurements.
It requires the existing Test 1, 2, 4, 5, and 7 sections, and will fail
with a clear message if any are absent. Test 6 is included only if its data
is stored in the unified JSON.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, KeepTogether,
    )
except ImportError as exc:
    raise SystemExit("Install PDF support: python -m pip install 'reportlab>=4,<5'") from exc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
ASSETS = ROOT / 'assets'
METRICS = RESULTS / 'benchmark_results.json'
PDF = RESULTS / 'technical_report.pdf'


def pct(n):
    return f'{100 * float(n):.1f}%'


def num(n, precision=3):
    return f'{float(n):.{precision}f}'


def build_report():
    if not METRICS.exists():
        raise SystemExit(f'Missing results: {METRICS}. Run the benchmark stages first.')
    data = json.loads(METRICS.read_text())
    required = ['test1_prediction', 'test2_planning', 'test4_uncertainty',
                'test5_risk_aware_stress', 'test7_transformer_vs_ensemble']
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit('Missing benchmark sections: ' + ', '.join(missing)
                           + '. Do not publish an incomplete report.')

    palette = {
        'ink': colors.HexColor('#17293c'),
        'muted': colors.HexColor('#56677b'),
        'line': colors.HexColor('#dbe5ed'),
        'accent': colors.HexColor('#087ea4'),
        'pale': colors.HexColor('#eef6f9'),
    }
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name='ReportTitle', parent=ss['Title'], fontName='Helvetica-Bold',
                          fontSize=21, leading=26, textColor=palette['ink'], spaceAfter=10))
    ss.add(ParagraphStyle(name='ReportSubtitle', parent=ss['Normal'],
                          fontSize=11, leading=16, textColor=palette['muted'], spaceAfter=20))
    ss.add(ParagraphStyle(name='ReportH1', parent=ss['Heading1'],
                          fontSize=15, leading=19, spaceBefore=16, spaceAfter=8,
                          textColor=palette['ink']))
    ss.add(ParagraphStyle(name='ReportH2', parent=ss['Heading2'],
                          fontSize=11, leading=15, spaceBefore=11, spaceAfter=6,
                          textColor=palette['accent']))
    ss.add(ParagraphStyle(name='ReportBody', parent=ss['BodyText'],
                          fontSize=9, leading=14, spaceAfter=8, textColor=palette['ink']))
    ss.add(ParagraphStyle(name='ReportNote', parent=ss['BodyText'],
                          fontSize=8, leading=11, spaceAfter=8, textColor=palette['muted']))
    ss.add(ParagraphStyle(name='ReportCaption', parent=ss['BodyText'],
                          fontSize=8, leading=11, alignment=TA_CENTER,
                          textColor=palette['muted'], spaceAfter=8))
    ss.add(ParagraphStyle(name='Cell', parent=ss['BodyText'],
                          fontSize=8, leading=11, textColor=palette['ink']))
    ss.add(ParagraphStyle(name='CellHeader', parent=ss['BodyText'],
                          fontName='Helvetica-Bold', fontSize=8, leading=11,
                          textColor=colors.white))

    story = []

    def P(t, style='ReportBody'):
        story.append(Paragraph(t, ss[style]))

    def section(name):
        P(name, 'ReportH1')

    def sub(name):
        P(name, 'ReportH2')

    def tab(rows, widths=None):
        flow = [[Paragraph(escape(str(c)), ss['CellHeader' if i == 0 else 'Cell'])
                 for c in row] for i, row in enumerate(rows)]
        table = Table(flow, colWidths=widths, repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), palette['ink']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, palette['pale']]),
            ('BOX', (0, 0), (-1, -1), 0.3, palette['line']),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, palette['line']),
            ('LEFTPADDING', (0, 0), (-1, -1), 9),
            ('RIGHTPADDING', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(table)
        story.append(Spacer(1, 9))

    def fig(name, caption, max_w=6.9 * inch, max_h=3.55 * inch):
        path = ASSETS / name
        if not path.exists():
            P(f'Figure unavailable in repository: {escape(name)}', 'ReportNote')
            return
        img = Image(str(path))
        factor = min(max_w / img.imageWidth, max_h / img.imageHeight)
        img.drawWidth = img.imageWidth * factor
        img.drawHeight = img.imageHeight * factor
        story.append(img)
        P(escape(caption), 'ReportCaption')

    def new_page():
        story.append(PageBreak())

    def footer(canvas, doc):
        canvas.saveState()
        w, _ = doc.pagesize
        canvas.setStrokeColor(palette['line'])
        canvas.line(42, 43, w - 42, 43)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(palette['muted'])
        canvas.drawString(43, 30, 'Predictive Robot Intelligence | Synthetic 2D simulation research')
        canvas.drawRightString(w - 43, 30, f'Page {doc.page}')
        canvas.restoreState()

    t1 = data['test1_prediction']
    ti, to = t1['in_distribution'], t1['out_of_distribution']
    planning = data['test2_planning']
    corr = data['test4_uncertainty']['disagreement_error_correlation']
    raw5 = data['test5_risk_aware_stress']
    # Stage 3 keys contain descriptive display labels with line breaks.
    p_mean = next((v for k, v in raw5.items() if 'mean' in k.lower()), None)
    p_risk = next((v for k, v in raw5.items() if 'cvar' in k.lower()), None)
    if p_mean is None or p_risk is None:
        raise SystemExit('Test 5 lacks ensemble mean or CVaR result. Regenerate Stage 3.')
    t7 = data['test7_transformer_vs_ensemble']['prediction_3s']

    # Page 1 - summary and scope
    P('Neural World Model &amp; Predictive Robot Intelligence', 'ReportTitle')
    P('Technical report | Learned dynamics, obstacle forecasting, multi-future MPC',
      'ReportSubtitle')
    P('<b>Project:</b> CPU-based learned world models for mobile-robot navigation '
      'among moving obstacles. <b>Scope:</b> synthetic 2D simulation; '
      'not Gazebo/Isaac Sim or physical-robot validation.')
    section('Abstract')
    P('A robot-dynamics MLP and a rotation-invariant obstacle-motion predictor are '
      'trained on simulated experience and used inside sampling-based model-predictive '
      'control. A five-member bootstrapped ensemble supports multiple possible '
      'obstacle futures and a risk-sensitive CVaR planner. In the 25-episode-per-planner '
      'stress evaluation reported here, ensemble-mean Neural-MPC achieved '
      f'{pct(p_mean["success_rate"])} success and {pct(p_mean["collision_rate"])} '
      'observed collision frequency, while Risk-MPC achieved '
      f'{pct(p_risk["success_rate"])} success and {pct(p_risk["collision_rate"])} '
      'observed collision frequency. These are measured outcomes from a small '
      'synthetic benchmark, not guarantees of robot safety.')
    section('System and methodology')
    P('A unicycle robot navigates a 20 m x 20 m simulated arena with moving circular '
      'obstacles, curved/noisy motions and wall interactions. The learned robot '
      'dynamics and history-conditioned obstacle model provide state rollouts. '
      'Sampling-MPC evaluates candidate action sequences using collision, '
      'proximity, goal, energy and smoothness penalties. Risk-MPC compares '
      'candidate actions across the ensemble and emphasizes higher-cost futures.')
    fig('architecture.png', 'Architecture of learning, imagined futures and planning.', max_h=2.8 * inch)
    P('<b>Evidence source:</b> results/benchmark_results.json and repository-generated '
      'figures. This PDF is a regenerated presentation of stored results, not a '
      'new training run.', 'ReportNote')
    new_page()

    # Page 2 - obstacle prediction
    section('Test 1 | Dynamic obstacle forecasting')
    P('The baseline compares constant-velocity extrapolation against the learned '
      'obstacle-motion model over a 3-second prediction horizon. Lower ADE/FDE is better.')
    tab([
        ['Predictor', 'ADE in [m]', 'FDE in [m]', 'ADE unseen [m]', 'FDE unseen [m]'],
        ['Constant velocity', num(ti['ADE_const_vel']), num(ti['FDE_const_vel']),
         num(to['ADE_const_vel']), num(to['FDE_const_vel'])],
        ['Neural obstacle model', num(ti['ADE_neural']), num(ti['FDE_neural']),
         num(to['ADE_neural']), num(to['FDE_neural'])],
    ], [2.0 * inch, 1.04 * inch, 1.04 * inch, 1.25 * inch, 1.25 * inch])
    fig('prediction_vs_groundtruth.png',
        'Example of forecasted obstacle paths versus ground truth.', max_h=3.4 * inch)
    P('Interpretation: The learned predictor reduces prediction error in both '
      'reported evaluation environments. Error around wall bounces and distribution '
      'shift remains a limitation. Obstacle positions are supplied by simulation '
      'rather than estimated from real sensors.', 'ReportNote')
    new_page()

    # Page 3 - navigation planning
    section('Tests 2 and 3 | Planning and generalization')
    P('The base environment and a harder unseen environment each evaluate '
      'the reactive, DWA-style constant-control, analytic-MPPI and learned-model '
      'Neural-MPC planners. Success and collision are empirical episode fractions.')
    rows = [['Planner', 'Success: base', 'Collision: base',
             'Success: unseen', 'Collision: unseen']]
    labels = [('Reactive','Reactive'),('DWA','DWA-style'),('MPPI','Analytic MPPI'),
              ('Neural-MPC','Neural-MPC')]
    for key, label in labels:
        a = planning['in_distribution'][key]
        b = planning['out_of_distribution'][key]
        rows.append([label, pct(a['success_rate']), pct(a['collision_rate']),
                     pct(b['success_rate']), pct(b['collision_rate'])])
    tab(rows, [1.8 * inch, 1.18 * inch, 1.2 * inch, 1.22 * inch, 1.22 * inch])
    fig('benchmark_planning.png', 'Base-environment planning benchmark.', max_h=2.7 * inch)
    P('For the analytic MPPI versus Neural-MPC ablation, the optimizer and '
      'cost terms are shared, while the dynamics and obstacle prediction model '
      'change. Reactive and DWA-style planners are different algorithms, so '
      'claiming identical cost across all four is not justified.', 'ReportNote')
    new_page()

    # Page 4 - uncertainty and stress benchmark
    section('Tests 4 and 5 | Model disagreement and risk-aware planning')
    sub('Test 4 | Empirical disagreement-error association')
    P(f'Across the reported 720 obstacle tracks, ensemble disagreement has a '
      f'correlation of r = {float(corr):.3f} with realized prediction error. '
      'This is a positive association, not a formal uncertainty-calibration test.')
    fig('uncertainty_ensemble.png',
        'Five predicted futures and empirical disagreement versus error.', max_h=2.7 * inch)
    sub('Test 5 | Stress evaluation with 12 moving obstacles')
    tab([
        ['Planner', 'Success', 'Collision', 'Mean energy'],
        ['Ensemble-mean Neural-MPC', pct(p_mean['success_rate']),
         pct(p_mean['collision_rate']), num(p_mean['energy'],2)],
        ['Risk-MPC (ensemble CVaR)', pct(p_risk['success_rate']),
         pct(p_risk['collision_rate']), num(p_risk['energy'],2)],
    ], [2.8 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch])
    P('These two planner results came from 25 simulated episodes each, '
      'as verified in the local Stage 2 console log. The unified JSON '
      'currently does not store that episode count. No single-model '
      'baseline was evaluated in this updated run.', 'ReportNote')
    new_page()

    # Page 5 - optional test 6, test7, limitations
    section('Test 7 | Transformer rollout stability')
    P('A sequence Transformer and a rotation-invariant MLP ensemble were '
      'compared on long-horizon prediction in the reported simulation.')
    tab([
        ['Predictor', '3-second ADE [m]', '3-second FDE [m]'],
        ['MLP ensemble', num(t7['ADE_ens']), num(t7['FDE_ens'])],
        ['Transformer', num(t7['ADE_tf']), num(t7['FDE_tf'])],
        ['Constant velocity', num(t7['ADE_cv']), num(t7['FDE_cv'])],
    ], [2.9 * inch, 1.75 * inch, 1.75 * inch])
    P('The preliminary Transformer experiment shows substantial autoregressive '
      'drift. The original evaluation also has methodological limitations '
      '(including sequence-window splitting and different input/rollout conventions), '
      'so the numbers should not be presented as a definitive architecture comparison.',
      'ReportNote')
    fig('autoregressive_drift.png',
        'Saved model-prediction drift visualization.', max_h=2.4 * inch)
    if 'test6_occupancy' not in data:
        sub('Test 6 | Occupancy prediction')
        P('The repository contains occupancy figures and code, but this unified '
          'benchmark JSON does not contain Test 6 metrics. This report deliberately '
          'does not reproduce unverified numerical claims for Test 6.', 'ReportNote')
    new_page()

    section('Limitations and reproducibility')
    P('<b>Simulation scope.</b> The environment is a compact 2D simulator with '
      'tracked obstacle coordinates and a simplified robot model. The work has not '
      'established end-to-end real-robot perception or safe physical deployment.')
    P('<b>Metrics.</b> All success and collision rates are observed within the '
      'specified simulation episodes; zero observed collisions does not imply '
      'zero collision probability. Stress results use small sample sizes, and '
      'comparisons with different evaluation seeds or model checkpoints require care.')
    P('<b>Model limitations.</b> The obstacle model relies on ordered, known '
      'obstacle tracks. It has limited treatment of interactions, identity switches, '
      'uncertain state estimates, and physical safety layers.')
    P('<b>Reproduction.</b> In a clean checkout, install requirements with '
      'the tested CPU constraints, run the base model/pipeline, then the ensemble '
      'upgrade stages. The base pipeline merges its own results into the unified '
      'JSON; historical sections should not be taken as results of a single '
      'fresh training run unless they have been reproduced together.')
    P('<b>Hardware transfer.</b> ROS 2 integration is an experimental scaffold. '
      'Gazebo, Isaac Sim, sensor integration and hardware safety validation remain '
      'future tasks. The 2D simulator is not a substitute for those tests.')
    sub('Data and provenance')
    P('Benchmark metrics: results/benchmark_results.json. Figures: assets/. '
      'Source scripts: scripts/run_pipeline.py, scripts/run_upgrade.py, '
      'scripts/run_stage2_benchmark.py. All metrics in the tables above are read '
      'from the supplied JSON; the Test 5 episode count is based on the '
      'separately recorded local Stage 2 run.', 'ReportNote')
    P(f'Generated: {date.today().isoformat()} | Document generated by scripts/make_technical_report.py',
      'ReportNote')

    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=(595.3, 841.9),
                            leftMargin=46, rightMargin=46,
                            topMargin=48, bottomMargin=58,
                            title='Neural World Model & Predictive Robot Intelligence - Technical Report',
                            author='Harshavardhan Sekar')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'[report] generated {PDF} ({PDF.stat().st_size} bytes)')


if __name__ == '__main__':
    build_report()
