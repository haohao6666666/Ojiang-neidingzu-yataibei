#!/usr/bin/env python3
"""English style linter for modelling-contest papers (MCM/ICM, APMCM).

    python lint_english.py sections/*.tex [--max-words 35] [--quiet-info]

Works on .tex / .md / .txt. LaTeX commands, math and comments are stripped before checking.
Every hit is a prompt to look, not an order to replace: judge each one in context.

Checks
  CJK         Chinese characters / full-width punctuation (APMCM: disqualifying)
  LONG        sentences longer than --max-words (O-award median is 23 words)
  CONTRACT    contractions (can't, it's ...)
  PHRASE      Chinglish / filler / AI-style phrases with suggested fixes
  CLAIM       over-claim words (prove, guarantee, optimal, always ...)
  SIGNIF      "significant(ly)" with no p-value / CI / test nearby
  SYMSTART    sentence starting with a math symbol
  ASSUME      "assume"/"suppose" not followed by "that"
  NUMWORD     small numbers (2-9) written as digits before a plain noun
  OPENING     three consecutive sentences starting with the same word
  FIGREF      "As shown in Figure" used more than 3 times
  PARA        paragraphs longer than 180 words
  FIGNAME     mixing "Fig." and "Figure"
Summary statistics (sentence length, 'we' rate, passive rate) are printed at the end.
"""
import argparse
import re
import statistics
import sys

PHRASES = [
    (r"\bobviously\b|\bit is obvious that\b|\bas we all know\b|\bit is well known that\b", "delete, or give the reason"),
    (r"\bit is worth (?:noting|mentioning) that\b|\bit should be (?:noted|mentioned|pointed out) that\b|\bnote that\b", "just state the point"),
    (r"\bplays? an? (?:important|vital|key|crucial|pivotal|significant) role\b", "say what the role is and how large"),
    (r"\bwith the (?:rapid )?development of\b", "delete; start with the concrete fact"),
    (r"\bmore and more\b", "increasingly / give the trend"),
    (r"\ba lot of\b|\blots of\b", "many / much / a number"),
    (r"\bvery\b", "use a precise word or a number"),
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b|\bowing to the fact that\b", "because / since"),
    (r"\bmake an? (?:analysis|investigation|comparison|prediction|evaluation) (?:of|on)\b|\bcarry out (?:an? )?(?:analysis|investigation)\b", "use the verb: analyse / compare / predict"),
    (r"\bfirstly\b", "First"),
    (r"\bin this paper,? we will\b|\bthis paper will\b", "We ..."),
    (r"\b(?:has|have|achieves?|obtains?) (?:a )?(?:good|better|great|certain) (?:effect|result|performance)s?\b", "quantify: reduces RMSE by x%"),
    (r"\ba certain\b", "be specific or delete"),
    (r"\bdatas\b", "data"),
    (r"\bthe influence of [^.]{1,40}? to\b", "the effect of X on Y"),
    (r"\bfrom (?:this|which) we can (?:know|see|get)\b|\bwe can know\b", "This indicates that ... / Hence ..."),
    (r"\bto sum up\b|\bin a word\b|\ball in all\b", "In summary (or just state the conclusion)"),
    (r"\band so on\b|\betc\.", "list fully or use 'such as'"),
    (r"\butili[sz](?:e|es|ed|ing|ation)\b", "use"),
    (r"\bdelve[sd]? into\b|\bin today'?s (?:rapidly )?(?:evolving|changing)\b|\bseamless(?:ly)?\b|\bgroundbreaking\b|\bcutting-edge\b|\bparadigm shift\b|\bholistic\b|\btapestry\b|\bnavigat(?:e|ing) the complexit", "AI-style wording: replace with concrete content"),
    (r"\bharness(?:ing)? the power of\b|\bunlock(?:s|ing)? the potential\b|\bpave[sd]? the way\b", "AI-style wording: say what actually happens"),
    (r"\bon the other hand\b", "only after 'on the one hand'; otherwise 'In contrast' / 'However'"),
    (r"\bthe results? shows?\b", "check agreement: 'the results show' / 'the result shows'"),
]
CLAIMS = r"\bprove[sd]?\b|\bguarantee[sd]?\b|\balways\b|\bnever\b|\bperfect(?:ly)?\b|\boptimal\b|\bbest\b|\bcompletely\b|\bnovel\b|\bfirst (?:time|to)\b"
CONTRACTIONS = r"\b(?:can't|won't|don't|doesn't|isn't|aren't|wasn't|weren't|it's|let's|we're|we'll|we've|that's|there's|couldn't|shouldn't|wouldn't|hasn't|haven't|didn't)\b"
CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
NOUNS_AFTER_NUM = r"(?:models?|methods?|steps?|stages?|tasks?|problems?|parts?|types?|categories|factors?|indicators?|variables?|scenarios?|layers?|groups?|clusters?|phases?|components?|aspects?|levels?|questions?|sub-?problems?|kinds?|ways?|cases?)"


def strip_latex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)                                   # comments
    text = re.sub(r"\\begin\{(equation|align|gather|multline|displaymath|eqnarray|tikzpicture|lstlisting|verbatim|algorithmic|tabular|tabularx)\*?\}.*?\\end\{\1\*?\}", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " MATH ", text)
    text = re.sub(r"\\(?:cite|ref|cref|Cref|eqref|label|includegraphics|input|url|si|SI)\*?(?:\[[^\]]*\])?\{[^}]*\}(?:\{[^}]*\})?", " REF ", text)
    text = re.sub(r"\\(?:textbf|textit|emph|texttt|underline|textsc|mbox|text)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph|caption)\*?\{([^{}]*)\}", r"\n\n\1.\n\n", text)
    text = re.sub(r"\\item\b", "\n\n", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}]", " ", text)
    text = text.replace("~", " ").replace("``", '"').replace("''", '"')
    return text


def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(par):
    par = re.sub(r"\s+", " ", par)
    par = re.sub(r"\b(e\.g|i\.e|et al|Fig|Eq|Eqs|vs|approx|resp|cf|Sec|No)\.", lambda m: m.group(0).replace(".", "<DOT>"), par)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"(MATH])", par)
    return [s.replace("<DOT>", ".").strip() for s in sents if len(s.split()) >= 3]


def line_of(raw, snippet):
    """Best-effort source line: search the longest plain-text chunk (no MATH/REF tokens)."""
    chunks = sorted((c.strip() for c in re.split(r"MATH|REF", snippet)), key=len, reverse=True)
    for c in chunks:
        words = c.split()
        for n in (6, 4, 3, 2):
            if len(words) >= n:
                pat = r"\s+".join(re.escape(w) for w in words[:n])
                m = re.search(pat, raw)
                if m:
                    return raw.count("\n", 0, m.start()) + 1
    return "?"


def lint(path, max_words, out):
    raw = open(path, encoding="utf-8", errors="replace").read()
    for i, ln in enumerate(raw.splitlines(), 1):
        code = re.sub(r"(?<!\\)%.*", "", ln)
        if CJK.search(code):
            out.append((path, i, "CJK", f"Chinese/full-width characters: {code.strip()[:50]!r}"))
    text = strip_latex(raw) if path.endswith(".tex") else raw
    stats = {"sent": [], "we": 0, "passive": 0, "words": 0}
    fig_shown = 0
    fig_short = bool(re.search(r"\bFig\.\s*(?:\d|REF)", text))
    fig_long = bool(re.search(r"\bFigure\s*(?:\d|REF)", text))
    for par in split_paragraphs(text):
        nwords = len(par.split())
        if nwords > 180:
            out.append((path, line_of(raw, par[:40]), "PARA", f"paragraph of {nwords} words - split by topic"))
        sents = split_sentences(par)
        openings = []
        for s in sents:
            w = len(s.split())
            stats["sent"].append(w)
            stats["words"] += w
            low = s.lower()
            stats["we"] += len(re.findall(r"\bwe\b", low))
            stats["passive"] += len(re.findall(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", low))
            ln = line_of(raw, s)
            if w > max_words:
                out.append((path, ln, "LONG", f"{w} words: {s[:70]}..."))
            if re.search(CONTRACTIONS, s, re.I):
                out.append((path, ln, "CONTRACT", re.search(CONTRACTIONS, s, re.I).group(0)))
            for pat, fix in PHRASES:
                m = re.search(pat, s, re.I)
                if m:
                    out.append((path, ln, "PHRASE", f"'{m.group(0)}' -> {fix}"))
            m = re.search(CLAIMS, s, re.I)
            if m and not re.search(r"pareto[- ]optimal|best (?:among|of the)|optimal (?:solution|value) (?:of|to) (?:the|eq)", s, re.I):
                out.append((path, ln, "CLAIM", f"'{m.group(0)}' - is there proof/evidence for this strength?"))
            if re.search(r"\bsignificant(?:ly)?\b", s, re.I) and not re.search(r"\bp\s*[<=>]|p-value|confidence|\bCI\b|test|MATH", s, re.I):
                out.append((path, ln, "SIGNIF", "'significant' without a test/p/CI in the sentence - add evidence or use 'substantial' + number"))
            if s.startswith("MATH"):
                out.append((path, ln, "SYMSTART", f"sentence starts with a symbol: {s[:50]}"))
            if re.search(r"\b(?:assume|suppose)[sd]?\s+(?!that\b)(?:MATH|[a-z]\b)", s, re.I):
                out.append((path, ln, "ASSUME", "add 'that' after assume/suppose"))
            m = re.search(rf"\b([2-9])\s+(?:main\s+|key\s+|different\s+)?{NOUNS_AFTER_NUM}\b", s)
            if m:
                out.append((path, ln, "NUMWORD", f"'{m.group(0)}' - write small numbers as words (three models)"))
            if re.search(r"\bas (?:is )?(?:shown|illustrated|depicted|presented) in (?:Fig|Figure|Table|REF)", s, re.I):
                fig_shown += 1
            openings.append(s.split()[0].lower())
        for k in range(len(openings) - 2):
            if openings[k] == openings[k + 1] == openings[k + 2] and openings[k] not in ("MATH",):
                out.append((path, line_of(raw, sents[k]), "OPENING", f"three sentences in a row start with '{openings[k]}'"))
                break
    if fig_shown > 3:
        out.append((path, "-", "FIGREF", f"'As shown in Figure/Table' used {fig_shown} times - let the finding be the subject"))
    if fig_short and fig_long:
        out.append((path, "-", "FIGNAME", "both 'Fig.' and 'Figure' used - pick one"))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max-words", type=int, default=35)
    a = ap.parse_args()
    out, allsent, we, passive, words = [], [], 0, 0, 0
    for f in a.files:
        st = lint(f, a.max_words, out)
        allsent += st["sent"]; we += st["we"]; passive += st["passive"]; words += st["words"]
    order = ["CJK", "CLAIM", "SIGNIF", "PHRASE", "CONTRACT", "SYMSTART", "ASSUME", "NUMWORD", "LONG", "PARA", "OPENING", "FIGREF", "FIGNAME"]
    out.sort(key=lambda r: (order.index(r[2]), str(r[0]), str(r[1])))
    for f, ln, kind, msg in out:
        print(f"{f}:{ln}: [{kind}] {msg}")
    if allsent:
        print("\n--- summary ---")
        print(f"sentences {len(allsent)}, words {words}, median length {statistics.median(allsent):.0f} "
              f"(O-award corpus: 23), share > 35 words {sum(s > 35 for s in allsent) / len(allsent):.0%}")
        print(f"'we' per 10k words {we * 1e4 / max(words, 1):.0f} (corpus: ~75); "
              f"passive per 10k words {passive * 1e4 / max(words, 1):.0f} (corpus: ~48)")
    print(f"{len(out)} findings")
    sys.exit(1 if any(r[2] == "CJK" for r in out) else 0)


if __name__ == "__main__":
    main()
