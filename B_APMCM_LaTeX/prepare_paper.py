import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOK = "B题论文文稿.md"
A = "B_results/figures"
FIGURES = {
    1: ("fig01_dual_modal_schematic", 0.75),
    2: ("fig02_framework", 1.0),
    3: ("fig03_preprocessing_workflow", 1.0),
    4: ("fig04_o2_counter_beat", 1.0),
    5: ("fig05_carbon_oxygen_balance", 0.7),
    6: ("fig06_arrival_time_map", 0.8),
    7: ("fig07_rlr_decomposition", 1.0),
    8: ("fig08_outlier_reconstruction", 1.0),
    9: ("fig09_spectral_surfaces", 1.0),
    10: ("fig10_process_curves_heat03", 1.0),
    11: ("fig11_four_stage_relation", 1.0),
    12: ("fig12_plug_flow_delay", 1.0),
    13: ("fig13_windowed_delay_vs_flow", 0.7),
    14: ("fig14_loho_predictions", 1.0),
    15: ("fig15_transfer_probe_vip", 1.0),
}
CONTENTS_TITLES = {
    "3": "Data Preprocessing and Consistency",
    "3.1": "Converter Groups and Wavelength Grid",
    "3.2": "Attachment 1 and the Heat 5 Time Step",
    "5": "Task 1: Spectral Reconstruction",
    "6": "Task 2: Process Signals and Variable Delay",
    "7": "Task 3: Real-Time Off-Gas Prediction",
    "9": "Model Evaluation and Discussion",
}
FIGURE_LEAD_INS = {
    1: r"Figure~\ref{fig:1} shows the two measurement locations.",
    9: r"\ifpaperappendix Appendix~\ref{app:heat-figures} contains the individual spectral surfaces.\fi",
    10: r"\ifpaperappendix The remaining nine process plots are in Appendix~\ref{app:heat-figures}.\fi",
    11: r"Figure~\ref{fig:11} shows the piecewise fits for the ten heats.",
}
PREPROCESSING_CAPTION = "Five-step data preprocessing and consistency workflow."
LOHO_ENDPOINT_CAPTION = ""
ESCAPE = str.maketrans({"#": r"\#", "%": r"\%", "&": r"\&", "_": r"\_", "~": r"\textasciitilde{}",
                        "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}", "^": r"\textasciicircum{}",
                        "−": r"\ensuremath{-}", "±": r"\ensuremath{\pm}",
                        "×": r"\ensuremath{\times}", "≤": r"\ensuremath{\leq}",
                        "≥": r"\ensuremath{\geq}", "≈": r"\ensuremath{\approx}",
                        "→": r"\ensuremath{\to}", "∞": r"\ensuremath{\infty}"})
GREEK = dict(zip("αβγδΔητλμρσφπ", ("alpha","beta","gamma","delta","Delta","eta","tau","lambda",
                                             "mu","rho","sigma","varphi","pi")))
TOKEN = re.compile(r"\$[^$\n]+\$|\*\*[^*\n]+\*\*|(?<!\*)\*[^*\n]+\*|" +
                   chr(96) + r"[^" + chr(96) + r"\n]+" + chr(96) +
                   r"|\b(?:Equation\s+\([0-9]+\)|(?:Figures?|Tables?|Equations?|Sections?)\s+[0-9]+(?:\.[0-9]+)*)"
                   r"|\[(?:\d{1,2})(?:\s*[,–-]\s*\d{1,2})*\]|[₂₁₀]+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+|\b(?:Appendix [ABC])\b|[αβγδΔητλμρσφπ]")
SCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻₀₁₂", "0123456789-012")


def plain(value):
    return value.translate(ESCAPE)


def markup(value, *, references=True):
    out = []
    end = 0
    for match in TOKEN.finditer(value):
        out.append(plain(value[end:match.start()]))
        s = match.group()
        if s.startswith("$"):
            out.append(s.replace("−", "-").replace("–", "-"))
        elif s.startswith("**"):
            out.append(r"\textbf{" + markup(s[2:-2], references=references) + "}")
        elif s.startswith("*"):
            out.append(r"\emph{" + markup(s[1:-1], references=references) + "}")
        elif s.startswith(chr(96)):
            out.append(r"\texttt{" + plain(s[1:-1]) + "}")
        elif s in GREEK:
            out.append(r"\ensuremath{\%s}" % GREEK[s])
        elif s[0] in "⁰¹²³⁴⁵⁶⁷⁸⁹⁻₀₁₂":
            kind = "_" if s[0] in "₀₁₂" else "^"
            out.append(r"\ensuremath{%s{%s}}" % (kind, s.translate(SCRIPTS)))
        elif s.startswith("[") and references:
            a = re.findall(r"\d+", s)
            if ("–" in s or "-" in s) and len(a) == 2:
                a = list(map(str, range(int(a[0]), int(a[1]) + 1)))
            out.append(r"\cite{" + ",".join("ref:" + n for n in a) + "}")
        elif references and re.match(r"^(Figure|Table|Equation|Section) ", s):
            what, n = s.rstrip(")").split(None, 1)
            n = n.lstrip("(")
            prefix = {"Figure": "fig", "Table": "tab", "Equation": "eq", "Section": "sec"}[what]
            ref = r"\ref{" + prefix + ":" + n + "}"
            out.append(what + "~" + ("(" + ref + ")" if prefix == "eq" else ref))
        else:
            out.append(plain(s))
        end = match.end()
    out.append(plain(value[end:]))
    return "".join(out)


def figure(number, caption):
    stem, width = FIGURES[number]
    placement = "!htbp"
    size = str(width) + r"\linewidth"
    if number == 3:
        caption = PREPROCESSING_CAPTION
    if number == 14:
        caption += LOHO_ENDPOINT_CAPTION
    if number == 15:
        caption = "Cross-heat transfer, probe-saving trade-off and wavelength importance."
        placement = "!p"
        size = r"\linewidth,height=.92\textheight,keepaspectratio"
    graphic = r"\includegraphics[width=" + size + "]{figures/" + stem + ".pdf}"
    return "\n".join([r"\begin{figure}[" + placement + "]", r"\centering", graphic,
                      r"\caption{" + markup(caption) + r"}\label{fig:" + str(number) + "}", r"\end{figure}"])


def render_table(number, caption, lines):
    layouts = {
        1: "c Z c", 2: r"c c c c K{.09\linewidth} K{.13\linewidth} K{.13\linewidth} Z",
        3: r"Z K{.13\linewidth} K{.13\linewidth} K{.10\linewidth} K{.10\linewidth} K{.18\linewidth}",
        4: "Z c c", 5: r"Z K{.24\linewidth} K{.22\linewidth} Z",
        6: "Z Z Z", 7: r"Z K{.26\linewidth} Z",
        8: r"Z K{.15\linewidth} c c c c", 9: r"K{.07\linewidth} K{.20\linewidth} K{.21\linewidth} Z",
    }
    rows = [[x.strip() for x in line.strip().strip("|").split("|")] for line in lines]
    rows.pop(1)
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError((number, rows))
    if number == 6:
        rows = [list(row) for row in zip(*rows)]
    o = [r"\begin{table}[!htbp]", r"\centering\small",
         r"\caption{" + markup(caption) + r"}\label{tab:" + str(number) + "}",
         r"\renewcommand{\arraystretch}{1.12}",
         r"\begin{tabularx}{\linewidth}{@{}" + layouts[number] + r"@{}}", r"\toprule"]
    for i, row in enumerate(rows):
        cells = [markup(t) for t in row]
        if i == 0:
            cells = [r"\textbf{" + t + "}" for t in cells]
        o.append(" & ".join(cells) + r" \\")
        if i == 0:
            o.append(r"\midrule")
    o += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(o)


def render_body(lines):
    i = 0
    out = []
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        h = re.fullmatch(r"(#{2,4}) ([\d.]+) (.+)", line)
        f = re.match(r">\s*\*\*\[Figure (\d+)(?:[^\]]*)\]", line)
        t = re.match(r">\s*\*\*Table (\d+)\. (.+)\*\*$", line)
        if h:
            level, number, title = h.groups()
            command = ("section", "subsection", "subsubsection")[len(level) - 2]
            if command == "section" and out and number != "7":
                out.append(r"\FloatBarrier")
            short = "[" + markup(CONTENTS_TITLES[number]) + "]" if number in CONTENTS_TITLES else ""
            out.append("\\" + command + short + "{" + markup(title) + r"}\label{sec:" + number + "}")
        elif f:
            cap = lines[i].strip()
            assert cap.startswith("> Caption: "), (line, cap)
            number = int(f[1])
            if number in FIGURE_LEAD_INS:
                out.append(FIGURE_LEAD_INS[number])
            out.append(figure(number, cap.removeprefix("> Caption: ")))
            i += 1
        elif t:
            while i < len(lines) and not lines[i].strip():
                i += 1
            begin = i
            while i < len(lines) and lines[i].startswith("|"):
                i += 1
            number = int(t[1])
            if number == 1:
                out.append(r"Table~\ref{tab:1} lists the symbols and their units.")
            out.append(render_table(number, t[2], lines[begin:i]))
        elif line == "$$":
            begin = i
            while lines[i].strip() != "$$":
                i += 1
            equation = "\n".join(lines[begin:i])
            tag = re.search(r"\\tag\{(\d+)\}", equation)
            if tag:
                equation = equation.replace(tag[0], r"\label{eq:" + tag[1] + "}")
            env = "equation" if tag else "equation*"
            out.append("\\" + "begin{" + env + "}\n" + equation + "\n" + r"\end{" + env + "}")
            i += 1
        elif line.startswith("- "):
            out.append(r"\begin{itemize}")
            out.append(r"\item " + markup(line[2:]))
            while i < len(lines) and lines[i].startswith("- "):
                out.append(r"\item " + markup(lines[i][2:]))
                i += 1
            out.append(r"\end{itemize}")
        elif line.startswith(">") or line == "---":
            continue
        else:
            text = markup(line)
            out.append(text.replace(" (Appendix B)", r"\ifpaperappendix\ (Appendix~\ref{app:code})\fi"))
    return "\n".join(out) + "\n"


def render_references(lines):
    out = [r"\begin{thebibliography}{99}"]
    for line in lines:
        entry = re.fullmatch(r"\[(\d+)\]\s+(.+)", line)
        if entry:
            n, value = entry.groups()
            kind = "M" if n in {"6", "9", "10"} else "C" if n in {"2", "7"} else "J"
            if kind == "M":
                # The source lists a book as a whole. Preserve its bibliographic details.
                value = re.sub(r"(\*[^*]+\*)(?=\.| \()", r"\1 [M]", value, count=1)
            else:
                value = re.sub(r"(\(\d{4}\)\. .+?)(?=\. (?:In \*|\*))", r"\1 [" + kind + "]", value, count=1)
            out.append(r"\bibitem{ref:" + n + "} " + markup(value, references=False))
    return "\n".join([*out, r"\end{thebibliography}", ""])


def appendix(repo):
    local = ROOT / "source-listings"
    local.mkdir(exist_ok=True)
    out = [
        r"\section{Per-Heat Figures}\label{app:heat-figures}",
        r"The spectral surfaces cover all ten heats. The Heat 3 process curves appear in Figure~\ref{fig:10}.",
    ]
    for category, prefix, wording, heats in [
            ("Spectral Trend Surfaces", "q1_surface_heat", "Continuous spectral trend surface", range(1, 11)),
            ("Process Curves", "q2_curves_heat", "Reconstructed process curves", (1, 2, *range(4, 11)))]:
        out.append(r"\subsection{" + category + "}")
        for heat in heats:
            filename = f"{prefix}{heat:02}.pdf"
            if not (ROOT / "figures" / filename).is_file():
                shutil.copy2(repo / A / filename, ROOT / "figures" / filename)
            out.append("\n".join([
                r"\begin{figure}[!htbp]", r"\centering",
                r"\includegraphics[width=.95\linewidth,height=.39\textheight,keepaspectratio]{figures/" + filename + "}",
                r"\caption{" + wording + f" for Heat {heat}." + "}", r"\end{figure}",
            ]))
        out.append(r"\FloatBarrier")
    out.extend([
        r"\clearpage",
        r"\section{Software and Source Code}\label{app:code}",
        "Software: Python 3.13, NumPy, SciPy, pandas, scikit-learn, XGBoost, LightGBM, Matplotlib, and python-calamine. "
        "The code is listed in execution-package groups. Non-ASCII characters in the source listing are displayed as Unicode escapes.",
        "",
    ])
    dirs = [
        ("Spectral reconstruction and supporting analyses", "task1", repo / "B_results/code"),
        ("Delay model and cross-heat prediction", "task23", repo / "solution_v4_verified/solution_code"),
    ]
    for heading, prefix, folder in dirs:
        out.append(r"\subsection{" + heading + "}")
        for item in sorted(folder.iterdir()):
            if not item.is_file() or item.suffix not in {".py", ".sh", ".txt"}:
                continue
            path = local / (prefix + "-" + item.name)
            rendered = item.read_text(encoding="utf-8-sig").encode("ascii", "backslashreplace").decode("ascii")
            path.write_text(rendered, encoding="ascii")
            out.extend([
                r"\subsubsection*{\texttt{" + plain(item.name) + "}}",
                r"\VerbatimInput[fontsize=\scriptsize,breaklines,breakanywhere,breakanywheresymbolpre={},numbers=left,numbersep=4pt]{source-listings/"
                + path.name + "}",
            ])
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    md = (repo / BOOK).read_text(encoding="utf-8-sig")
    lines = md.splitlines()
    bounds = [next(i for i, v in enumerate(lines) if v.startswith(prefix)) for prefix in (
        "# Seeing", "## Summary", "## Contents", "## 1 Introduction", "## References", "## Appendix")]
    title, s, c, b, r, a = bounds
    project_sections = ROOT / "sections"
    project_sections.mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    title_main, separator, title_detail = lines[title][2:].partition(": ")
    summary = [r"\PaperTitle{" + markup(title_main + separator.rstrip()) + "}{" + markup(title_detail) + "}"]
    for raw in lines[s+1:c]:
        if raw.startswith("**Keywords:**"):
            summary.append(r"\keywords{" + markup(raw.split("**", 2)[2].strip()) + "}")
        elif raw != "---":
            summary.append(markup(raw))
    for number, (name, _) in FIGURES.items():
        if number in {1, 6, 7, 14, 15}:
            continue  # Generated by the editable scripts in figure-source/.
        filename = name + ".pdf"
        shutil.copy2(repo / "paper_figures" / "out" / filename, ROOT / "figures" / filename)
    for name, contents in [
        ("summary", "\n".join(summary) + "\n"), ("body", render_body(lines[b:r])),
        ("references", render_references(lines[r:a])), ("appendix", appendix(repo)),
    ]:
        (project_sections / (name + ".tex")).write_text(contents, encoding="utf-8")


if __name__ == "__main__":
    main()

