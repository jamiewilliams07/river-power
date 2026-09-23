"""
make_summary.py - builds the two-page summary (report/summary.pdf) and the
Results section of README.md.

Usage:
    python3 report/make_summary.py        (run after 15_screening.py)

Page 1 covers validation (does SWOT measure the river accurately?); page 2
covers the hydropower screening. Every number is read from outputs/, so the
PDF and README always match the latest run; only the text block below is
written by hand. The repository link is taken from `git remote get-url
origin` and falls back to a placeholder when no remote is set.

Inputs:  swot_config.yaml, outputs/*, figures/*
Outputs: report/summary.pdf, README.md (between the results markers)
"""

import json
import os
import re
import subprocess
import sys

import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# ---------------------------------------------------------------- hand-written text
NAME = "Jamie Williams"
AFFIL = "Mechanical Engineering, Carleton University"
DATE = "September 2026"
REPO = "github.com/YOUR-USERNAME/river-power"      # used only when no git remote is set
SECOND_REACH = "82285000481"                        # independent validation reach, 40 km downstream
SECOND_LABEL = "Reach 40 km downstream"

TITLE = "Hydropower screening from SWOT satellite observations on the Athabasca River"

CROSS_VERSION = (
    "<b>Version consistency.</b> SWOT product version D (SWORD v17b) reuses some reach IDs "
    "for different locations, giving WSE differences of 13&ndash;20 m for the same pass. "
    "After matching reaches by coordinates, all 19 agree within 9 cm.")

LIMITS = [
    "Gross theoretical potential only: no costs, access, ice, fish passage or land use.",
    "The rating curve requires calibration against a gauge.",
    "Velocity depends on an assumed roughness coefficient; no velocity data were used.",
    "SWOT observations cover the open-water season (May&ndash;Oct) only.",
    "Two reaches are missing from the profile, so total head is understated.",
]

TAKEAWAY = (
    "<b>Conclusion.</b> On this river, SWOT WSE is a far better predictor of discharge than "
    "channel width, and SWOT slope alone locates the head. Next steps: extend the validation to "
    "other gauged Canadian rivers, and test CSA RADARSAT Constellation imagery on rivers narrower "
    "than SWOT's ~100 m limit.")
# -------------------------------------------------------------------------

# Colours and paragraph styles for the PDF
INK, MUTED, RULE = colors.HexColor("#1a1a1a"), colors.HexColor("#555555"), colors.HexColor("#c8c8c8")
base = ParagraphStyle("base", fontName="Helvetica", fontSize=8.4, leading=10.4, textColor=INK)
title_s = ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=12.5, leading=15)
byline = ParagraphStyle("byline", parent=base, fontSize=8, textColor=MUTED)
h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=10, leading=12,
                    spaceBefore=2, spaceAfter=3)
head = ParagraphStyle("head", parent=base, fontName="Helvetica-Bold", fontSize=9, leading=11,
                      spaceBefore=3, spaceAfter=2)
cap = ParagraphStyle("cap", parent=base, fontSize=7, leading=8.4, textColor=MUTED)
cell = ParagraphStyle("cell", parent=base, fontSize=7.8, leading=9.4)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
bullet = ParagraphStyle("bullet", parent=base, fontSize=7.8, leading=9.6, leftIndent=8)


def repo_url():
    """Repository address from the git remote, or the REPO placeholder if none is set."""
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        url = ""
    m = re.search(r"github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?/?$", url)
    return f"github.com/{m.group(1)}" if m else REPO


def plain(s):
    """Convert ReportLab markup to Markdown so the README can reuse the PDF text."""
    s = re.sub(r"</?b>", "**", str(s))
    s = s.replace("<super>", "<sup>").replace("</super>", "</sup>")
    return s.replace("|", "/")


def md_table(rows):
    out = ["| " + " | ".join(plain(c) for c in rows[0]) + " |",
           "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(plain(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def update_readme(block, path="README.md"):
    """Replace the text between the results markers in README.md."""
    if not os.path.exists(path):
        return False
    text = open(path).read()
    start, end = text.find("<!-- results:start"), text.find("<!-- results:end -->")
    if start < 0 or end < start:
        return False
    start = text.index("-->", start) + 3
    with open(path, "w") as f:
        f.write(text[:start] + "\n" + block.strip() + "\n" + text[end:])
    return True


def need(path):
    if not os.path.exists(path):
        sys.exit(f"Missing {path}. Run the analysis scripts (12-15) first.")
    return path


def fig(path, width):
    w, h = ImageReader(need(path)).getSize()
    return Image(path, width=width, height=width * h / w)


def val_summary(reach):
    path = f"outputs/validation_summary_{reach}.csv"
    if not os.path.exists(path):
        return None
    v = pd.read_csv(path)
    get = lambda key: v[v["result"].str.contains(key, regex=False)].iloc[0] \
        if v["result"].str.contains(key, regex=False).any() else None
    return {"r1": get("1 height"), "loo": get("height, leave-one-out"),
            "split": get("time split"), "width": get("2b")}


def fmt(x, spec, suffix=""):
    try:
        return f"{float(x):{spec}}{suffix}"
    except (TypeError, ValueError):
        return "&ndash;"


def table(rows, widths, bold_first_row=True):
    t = Table([[Paragraph(str(c), cellb if (i == 0 and bold_first_row) else cell)
                for c in r] for i, r in enumerate(rows)], colWidths=widths)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    return t


def fig_pair(left, right, cap_l, cap_r, W, max_w=None):
    half = (W - 10) / 2
    w = min(half, max_w) if max_w else half
    t = Table([[fig(left, w), fig(right, w)],
               [Paragraph(cap_l, cap), Paragraph(cap_r, cap)]], colWidths=[half, half])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def bullets(items):
    return [Paragraph(i, bullet, bulletText="•") for i in items]


def two_cols(left, right, W, split=0.56):
    t = Table([[left, right]], colWidths=[W * split, W * (1 - split)])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (0, 0), 10), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
    return t


def main():
    cfg = yaml.safe_load(open(need("swot_config.yaml")))
    gauge_reach = str(cfg.get("gauge_reach"))
    upstream = {str(r) for r in (cfg.get("upstream_of_tributary") or [])}
    scr = json.load(open(need("outputs/screening_summary.json")))
    sc = pd.read_csv(need("outputs/screening.csv"), dtype={"reach_id": str})
    exc = int(scr["exceedance_pct"])
    s_set = scr["settings"]

    v1, v2 = val_summary(gauge_reach), val_summary(SECOND_REACH)
    if v1 is None:
        sys.exit(f"Missing outputs/validation_summary_{gauge_reach}.csv -- run 13_validate.py")
    errs = [float(v["loo"]["median_abs_pct_err"]) for v in (v1, v2) if v is not None]
    err_txt = (f"{min(errs):.0f}-{max(errs):.0f}%" if len(errs) > 1 and round(min(errs)) != round(max(errs))
               else f"{errs[0]:.0f}%")
    n_val = "two independent reaches" if v2 is not None else "the gauge reach"
    w_rs = []
    for v in (v1, v2):
        try:
            w_rs.append(float(v["width"]["correlation"]))
        except (TypeError, KeyError, ValueError):
            pass
    w_rs = [r for r in w_rs if r == r]
    if w_rs and max(abs(r) for r in w_rs) < 0.3:
        width_txt = "SWOT width showed no usable relationship with discharge"
    elif w_rs:
        width_txt = f"SWOT width was also correlated with discharge (r up to {max(w_rs):.2f})"
    else:
        width_txt = "SWOT width was not tested"

    up_drop = float(sc.loc[sc["reach_id"].isin(upstream), "drop_m"].sum())
    n_up = int(sc["reach_id"].isin(upstream).sum())
    top = scr["shortlist"][0] if scr["shortlist"] else None
    max_diff = scr.get("max_power_diff_pct")

    drop_check = ""
    if os.path.exists("outputs/drop_check.csv"):
        dc = pd.read_csv("outputs/drop_check.csv")
        a, b = dc["from_slopes_m"].sum(), dc["from_heights_m"].sum()
        drop_check = f"{abs(100 * (a - b) / b):.1f}%"
    clean_txt = ""
    if os.path.exists("outputs/swot_cleaning_report.csv"):
        cr = pd.read_csv("outputs/swot_cleaning_report.csv")
        clean_txt = f"{int(cr['total'].sum()):,} observations, {int(cr['kept'].sum()):,} retained"

    summary = (
        "The gross hydropower potential of a river reach is P = &rho;gQH&eta;, which requires "
        "discharge Q and hydraulic head H. I tested whether SWOT satellite observations (NASA/CNES, "
        "with the Canadian Space Agency) can provide both along the Athabasca River at Fort "
        "McMurray, using Water Survey of Canada gauge 07DA001 as the reference. "
        f"<b>A rating curve fitted to SWOT water-surface elevation predicted gauged discharge with "
        f"a median error of {err_txt} under leave-one-out cross-validation at {n_val}</b>. {width_txt}. ")
    cal = {c["role"]: c for c in scr["calibrations"]}
    uc = scr.get("upstream_calibration")
    gradient_txt, limit_txt = "", ""
    if os.path.exists("outputs/upstream_calibration.csv"):
        ut = pd.read_csv("outputs/upstream_calibration.csv").sort_values("dist_out_km")
        errs = ut["loo_median_err_pct"].tolist()
        rs = ut["backwater_r"].tolist()
        if len(errs) >= 4:
            near, rest = errs[:2], errs[2:]
            if near[0] > 1.5 * max(rest[1:] or rest):
                gradient_txt = (f"Accuracy decreases near the Clearwater confluence: {near[0]:.0f}% "
                                f"and {near[1]:.0f}% median error at the two nearest reaches, compared with "
                                f"{min(rest):.0f}&ndash;{max(rest):.0f}% further upstream.")
                r_near = max(abs(r) for r in rs[:2] if r == r) if any(r == r for r in rs[:2]) else None
                bw = uc.get("error_link") if uc else None
                if bw == "backwater":
                    cause = "This is consistent with backwater from the Clearwater"
                elif r_near is not None:
                    cause = ("Backwater from the Clearwater is a plausible cause, but a test "
                             "for it was inconclusive")
                else:
                    cause = "The cause has not been identified"
                limit_txt = (f"SWOT-derived discharge is less accurate near the Clearwater confluence "
                             f"({near[0]:.0f}% and {near[1]:.0f}% error). {cause}.")
    uc = scr.get("upstream_calibration")
    g = cal.get("gauge section")
    if g is not None:
        summary += (f"Using SWOT for both discharge and head, the design flow is within "
                    f"{abs(g['q30_diff_pct']):.0f}% of the gauged value at the gauge reach")
        if uc and uc["n_sections"] > 1:
            summary += (f" and a median {uc['rest_median_abs_q30_diff_pct']:.0f}% across "
                        f"{uc['n_sections'] - 1} reaches upstream. {gradient_txt} ")
        else:
            summary += ". "
    summary += (f"SWOT slope places about {up_drop:.0f} m of head in the {n_up} reaches of rapids "
                "above Fort McMurray")
    if top:
        summary += (f". The screening shortlists {len(scr['shortlist'])} reaches, led by one "
                    f"{top['dist_out_km']:.0f} km from the river mouth "
                    f"({top['power_MW']:.0f} MW, {top['energy_GWh_per_yr']:.0f} GWh/yr gross theoretical "
                    f"potential).")
    else:
        summary += ". No reach met all screening criteria."

    out = "report/summary.pdf"
    os.makedirs("report", exist_ok=True)
    margin = 0.4 * inch
    doc = SimpleDocTemplate(out, pagesize=letter, leftMargin=margin, rightMargin=margin,
                            topMargin=0.4 * inch, bottomMargin=0.35 * inch,
                            title="SWOT hydropower screening, Athabasca River", author=NAME)
    W = letter[0] - 2 * margin - 12

    # ------------------------------------------------------------- page 1
    repo = repo_url()
    link = f'<a href="https://{repo}" color="#1f6fb4">{repo}</a>' if repo != REPO else repo
    story = [Paragraph(TITLE, title_s), Spacer(1, 2),
             Paragraph(f"{NAME} &middot; {AFFIL} &middot; {DATE} &middot; Code and data: {link}", byline),
             Spacer(1, 5), Paragraph(summary, base), Spacer(1, 6),
             Paragraph("1. Does SWOT measure the river accurately?", h2)]

    def col(v, key, field, spec, suffix=""):
        return fmt(v[key][field], spec, suffix) if v is not None and v[key] is not None else "&ndash;"

    def pair(v, key, f1, s1, f2, s2, suf2=""):
        return f"{col(v, key, f1, s1)} / {col(v, key, f2, s2, suf2)}" if v is not None else "&ndash;"

    def scatter(v):
        return fmt(float(v["r1"]["scatter_m"]) * 100, ".0f", " cm") if v and v["r1"] is not None else "&ndash;"

    cols = ["", "Gauge reach"] + ([SECOND_LABEL] if v2 is not None else [])
    vals = [v1] + ([v2] if v2 is not None else [])
    rows = [cols,
            ["SWOT observations matched to gauge days"] + [col(v, "loo", "n", ".0f") for v in vals],
            ["WSE vs gauge stage anomaly: correlation / scatter"] +
            [f"{col(v, 'r1', 'correlation', '.2f')} / {scatter(v)}" for v in vals],
            ["Discharge error, leave-one-out (median)"] +
            [col(v, "loo", "median_abs_pct_err", ".1f", "%") for v in vals],
            ["Discharge error, latest 30% of dates (median)"] +
            [col(v, "split", "median_abs_pct_err", ".1f", "%") for v in vals],
            ["Width vs discharge: correlation"] + [col(v, "width", "correlation", ".2f") for v in vals]]
    widths = [W * 0.52] + [W * 0.48 / (len(cols) - 1)] * (len(cols) - 1)
    val_rows = rows
    story += [table(rows, widths), Spacer(1, 6)]

    story.append(fig_pair(f"figures/result1_height_vs_level_{gauge_reach}.png",
                          f"figures/result2_predicted_vs_gauged_{gauge_reach}.png",
                          "(a) SWOT WSE anomaly vs gauge stage anomaly at the gauge reach.",
                          "(b) Discharge predicted from SWOT WSE (leave-one-out) vs gauged discharge.",
                          W, max_w=2.75 * inch))
    story.append(Spacer(1, 4))
    fd = Table([[fig("figures/result5_flow_duration.png", W)],
                [Paragraph(f"(c) Flow-duration curves from SWOT and the gauge on coincident dates at "
                           f"three reaches. The design flow is the discharge exceeded {exc}% of "
                           f"the time.", cap)]], colWidths=[W])
    fd.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story += [fd, Spacer(1, 4)]

    method = [
        "<b>Data.</b> SWOT River Single Pass observations (Jul 2023&ndash;Sep 2026, versions C and D) "
        f"from NASA's Hydrocron API for {len(cfg.get('reaches') or [])} reaches ({len(sc)} with "
        "sufficient valid observations); daily discharge and stage from Water Survey of Canada "
        f"gauges 07DA001 and 07CD001 (record from {scr['design_flow_long_record']['record_start'][:4]}).",
        f"<b>Quality control.</b> {clean_txt + '. ' if clean_txt else ''}Removed ice-season, "
        "flagged, partial, dark-water and duplicate observations, and outliers.",
        CROSS_VERSION,
        "<b>Rating curve.</b> Q = a(h &minus; h<sub>0</sub>)<super>b</super> fitted to SWOT WSE "
        "(0.8 &le; b &le; 5); evaluated by leave-one-out cross-validation and a chronological 70/30 split.",
    ]
    if drop_check:
        method.append(f"<b>Head.</b> SWOT slope &times; reach length, cross-checked against WSE "
                      f"differences between adjacent reaches: the two agree within {drop_check}.")
    story += [Paragraph("Method: measurement", head)] + bullets(method)

    # ------------------------------------------------------------- page 2
    story += [PageBreak(), Paragraph("2. Where is the hydropower potential?", h2)]
    story.append(fig_pair("figures/result3_profile.png", "figures/result6_velocity.png",
                          "(d) Water-surface profile from SWOT. Gaps indicate missing reaches.",
                          "(e) Mean velocity at the design flow from SWOT width and slope (Manning).", W))
    story.append(Spacer(1, 4))
    wide = 5.6 * inch
    ft = Table([[fig("figures/result7_screening.png", wide)],
                [Paragraph("(f) Annual energy per reach. Orange: shortlisted; blue: meets the "
                           "criteria; grey: insufficient head gradient or measurement confidence.", cap)]],
               colWidths=[wide])
    ft.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story += [ft, Spacer(1, 6)]

    short_rows = None
    if scr["shortlist"]:
        rows = [["Rank", "Reach (km from mouth)", "Head", "Head gradient", "Capacity",
                 "Annual energy", "Capacity factor", "Mean velocity"]]
        for i, s in enumerate(scr["shortlist"], 1):
            rows.append([f"#{i}", f"{s['reach_id']} ({s['dist_out_km']:.0f} km)",
                         f"{s['drop_m']:.1f} m", f"{s['head_gradient_m_per_km']:.2f} m/km",
                         f"{s['power_MW']:.0f} MW", f"{s['energy_GWh_per_yr']:.0f} GWh/yr",
                         f"{s['capacity_factor']:.2f}", f"{s['velocity_ms']:.1f} m/s"])
        short_rows = rows
        story += [Paragraph("Shortlist: reaches recommended for site assessment", head),
                  table(rows, [W * f for f in (0.06, 0.24, 0.09, 0.12, 0.11, 0.14, 0.13, 0.11)]),
                  Spacer(1, 3)]

    flow_txt = "<b>SWOT-only discharge.</b> Design flow, SWOT vs gauge (coincident dates): "
    if g is not None:
        flow_txt += (f"{g['q30_swot_same_dates']:.0f} vs {g['q30_gauge_same_dates']:.0f} "
                     f"m<super>3</super>/s ({g['q30_diff_pct']:+.1f}%) at the gauge reach")
    if uc:
        flow_txt += (f"; median error {uc['median_err_pct']:.0f}% across {uc['n_sections']} upstream "
                     f"reaches (07DA001 minus 07CD001). Power has the same relative error")
    flow_txt += "."
    vel = scr["velocity"]
    lo, mid, hi = vel["n_values"]
    screen_method = [
        flow_txt,
        f"<b>Power.</b> P = &rho;gQH&eta;; Q = design flow, exceeded "
        f"{exc}% of the time ({scr['design_flow_long_record']['main']:.0f} m<super>3</super>/s; "
        f"{scr['design_flow_long_record']['above_tributary']:.0f} m<super>3</super>/s above the "
        f"Clearwater); &eta; = {scr['efficiency']:.2f}.",
        f"<b>Annual energy.</b> Daily discharge, {scr['energy']['years_used']} years; "
        f"environmental flow {s_set['env_flow_fraction_of_mean']:.0%} of mean; "
        f"turbine off below {s_set['min_turbine_fraction']:.0%} of design flow. Capacity factor "
        f"= mean / installed output.",
        f"<b>Velocity.</b> Manning's equation (wide channel) with SWOT width and slope; "
        f"roughness coefficient n = {mid} (range {lo}&ndash;{hi}).",
        f"<b>Criteria.</b> Head gradient &ge; {s_set['min_head_gradient_m_per_km']} m/km; &ge; "
        f"{s_set['min_readings']} valid observations; head spread (P10&ndash;P90)/median &le; "
        f"{s_set['max_drop_spread']:.0%}. {scr['n_pass']} of {scr['n_sections']} reaches pass; "
        f"ranked by annual energy per km.",
    ]
    left = [Paragraph("Method: screening", head)] + bullets(screen_method)
    limits = list(LIMITS)
    if limit_txt:
        limits.insert(1, limit_txt)
    elif uc and uc["adjacent"]["loo_median_err_pct"] > 15:
        adj = uc["adjacent"]
        why = {"backwater": "consistent with backwater from the Clearwater",
               "gauge_difference": "possibly because the reference discharge is a difference of two gauges"
               }.get(uc.get("error_link"), "cause not identified")
        limits.insert(1, f"SWOT-derived discharge is least accurate at the Clearwater confluence "
                         f"({adj['loo_median_err_pct']:.0f}% error), {why}.")
    right = [Paragraph("Limitations", head)] + bullets(limits)
    story += [two_cols(left, right, W), Spacer(1, 5), Paragraph(TAKEAWAY, base)]

    doc.build(story)
    print(f"wrote {out}")
    if repo == REPO:
        print("  Link in the PDF is the placeholder (no GitHub remote found). "
              "After `gh repo create`, run this again.")
    else:
        print(f"  Link in the PDF: {repo}")

    # ------------------------------------------------------ README results
    # Same values as the PDF, written between the results markers in README.md
    md = ["This section is generated from `outputs/` by `report/make_summary.py`. The full "
          "write-up is in [report/summary.pdf](report/summary.pdf).", "",
          "### 1. Does SWOT measure the river accurately?", "",
          plain(f"A rating curve fitted to SWOT water-surface elevation predicted gauged discharge with a "
                f"median error of <b>{err_txt}</b> under leave-one-out cross-validation at {n_val}. {width_txt}."), "",
          md_table(val_rows), "",
          f'<img src="figures/result1_height_vs_level_{gauge_reach}.png" width="49%"> '
          f'<img src="figures/result2_predicted_vs_gauged_{gauge_reach}.png" width="49%">', "",
          "*Left: SWOT WSE anomaly vs gauge stage anomaly. Right: discharge predicted from SWOT "
          "WSE (leave-one-out) vs gauged discharge.*", ""]
    if clean_txt:
        md += [plain(f"Quality control: {clean_txt}. {CROSS_VERSION}"), ""]
    md += ["### 2. Where is the hydropower potential?", ""]
    if g is not None:
        line = (f"**SWOT-only discharge:** using SWOT for both discharge and head, the design "
                f"flow (discharge exceeded {exc}% of the time) is within "
                f"{abs(g['q30_diff_pct']):.0f}% of the gauged value at the gauge reach")
        if uc and uc["n_sections"] > 1:
            line += (f", and a median {uc['rest_median_abs_q30_diff_pct']:.0f}% across "
                     f"{uc['n_sections'] - 1} reaches upstream. {gradient_txt}")
        else:
            line += "."
        md += [plain(line), ""]
    drop_line = f"**Head:** about {up_drop:.0f} m in the {n_up} reaches of rapids above Fort McMurray"
    drop_line += (f". Two independent SWOT estimates of head agree within {drop_check}."
                  if drop_check else ".")
    md += [drop_line, "", '<img src="figures/result3_profile.png" width="80%">', ""]
    v_up, v_dn = vel.get("upstream_range_ms"), vel.get("downstream_range_ms")
    if v_up and v_dn:
        md += [f"**Mean velocity** at the design flow: {v_up[0]:.1f}&ndash;{v_up[1]:.1f} m/s in the "
               f"rapids and {v_dn[0]:.1f}&ndash;{v_dn[1]:.1f} m/s below Fort McMurray (Manning's equation).", ""]
    md += [f"**Screening:** {scr['n_pass']} of {scr['n_sections']} reaches meet the criteria (head "
           f"gradient &ge; {s_set['min_head_gradient_m_per_km']} m/km, &ge; {s_set['min_readings']} valid "
           f"observations, consistent SWOT head). Shortlist, ranked by annual energy per km:", ""]
    if short_rows:
        md += [md_table(short_rows), ""]
    md += ['<img src="figures/result7_screening.png" width="80%">', "",
           "*Annual energy per reach. Orange: shortlisted; blue: meets the criteria; grey: "
           "insufficient head gradient or measurement confidence.*", "",
           "All values are gross theoretical potential for screening, not plant designs "
           "(see Limitations)."]
    if update_readme("\n".join(md)):
        print("  Filled the Results section of README.md")


if __name__ == "__main__":
    main()
