"""Export main.tex to an IEEE Access Word file that uses the styles and two-column layout of Access-Template-2024.docx.
Usage (from this folder): python3 export_docx.py  ->  main.docx
Steps: resolve \\ref/\\cite numbers in LaTeX, convert the body with pandoc (equations become Word equations), then rebuild the
front matter and remap every paragraph to the template's styles with python-docx. Needs pandoc, pdftoppm and python-docx."""
import re, subprocess, copy, tempfile
from pathlib import Path
import docx, docx.enum.text
from docx.shared import Inches, Pt
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parents[2] / "Docs/details/IEEE Acess Q1/Access-Template-2024.docx"
TEX = (HERE / "main.tex").read_text()
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def arg(s, macro):
    m = re.search(r"\\" + macro + r"\{", s)
    i, depth = m.end(), 1
    j = i
    while depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i:j - 1]


tex = strip_comments(TEX)
title = arg(tex, "title")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
keywords = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", tex, re.S).group(1)
body = tex[tex.index("\\maketitle") + len("\\maketitle"):tex.index("\\begin{thebibliography}")]
bib = re.findall(r"\\bibitem\{([^}]*)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", tex, re.S)

# --- numbering for \ref and \eqref --------------------------------------------------------------
labels, sec, sub, fig, tab, eq, app, in_app, last = {}, 0, 0, 0, 0, 0, 0, False, None
for m in re.finditer(r"\\(appendices)\b|\\(section\*?|subsection|label)\{([^}]*)\}|\\begin\{(figure\*?|table\*?|equation)\}", body):
    kind = m.group(1) or m.group(2) or m.group(4)
    if kind == "appendices":
        in_app = True
    elif kind == "section":
        if in_app:
            app += 1; last = chr(64 + app)
        else:
            sec += 1; sub = 0; last = ROMAN[sec - 1]
    elif kind == "subsection":
        sub += 1; last = f"{ROMAN[sec - 1]}-{chr(64 + sub)}"
    elif kind in ("figure", "figure*"):
        fig += 1; last = str(fig)
    elif kind in ("table", "table*"):
        tab += 1; last = ROMAN[tab - 1]
    elif kind == "equation":
        eq += 1; last = str(eq)
    elif kind == "label":
        labels[m.group(3)] = last
bibnum = {k: i + 1 for i, (k, _) in enumerate(bib)}


def cite(m):
    ns = sorted(bibnum[k.strip()] for k in m.group(1).split(","))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(f"[{ns[i]}]--[{ns[j]}]" if j - i >= 2 else ", ".join(f"[{n}]" for n in ns[i:j + 1]))
        i = j + 1
    return ", ".join(out)


eqn = iter(range(1, 100))
body = re.sub(r"\\cite\{([^}]*)\}", cite, body)
body = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({labels[m.group(1)]})", body)
body = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], body)
body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}",
              lambda m: "\n\n$$" + re.sub(r"\\label\{[^}]*\}", "", m.group(1)).strip() + f"$$ QQEQ{next(eqn)}QQ\n\n", body, flags=re.S)
body = re.sub(r"\$_\{?([A-Za-z0-9]+)\}?\$", r"\\textsubscript{\1}", body)
body = re.sub(r"\$\^\\mathrm\{([a-z])\}\$", r"\\textsuperscript{\1}", body)
body = body.replace("$^\\ast$", "\\textsuperscript{*}")
body = re.sub(r"\\PARstart\{(.)\}\{([^}]*)\}", r"\1\2", body)

head, _, tail = body.partition("\\appendices")
app_titles = iter("ABCDEFG")
tail = re.sub(r"\\section\{(.*)\}", lambda m: "\\section*{Appendix " + next(app_titles) + ". " + m.group(1) + "}", tail)
body = re.sub(r"\\section\*\{(.*)\}", r"\\section*{QQUQQ\1}", head + tail)
body = body.replace(".pdf}", ".png}")
body = body.replace("\\begin{table*}", "\\begin{table}").replace("\\end{table*}", "\\end{table}")
refs = "\n\\section*{QQUQQReferences}\n" + "\n\n".join(f"QQRQQ {t.strip()}" for _, t in bib)
body = body + refs

# --- pandoc ------------------------------------------------------------------------------------
for pdf in (HERE / "figs").glob("*.pdf"):
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(pdf.with_suffix(""))], check=True)
with tempfile.TemporaryDirectory() as td:
    src = Path(td) / "body.tex"
    src.write_text("\\documentclass{article}\\usepackage{amsmath,graphicx}\\begin{document}\n" + body + "\n\\end{document}\n")
    raw = Path(td) / "raw.docx"
    subprocess.run(["pandoc", str(src), "-f", "latex", "-o", str(raw), "--reference-doc", str(TEMPLATE), "--resource-path", str(HERE)], check=True)
    D = docx.Document(str(raw))

T = docx.Document(str(TEMPLATE))
tpl_sect = [copy.deepcopy(s._sectPr) for s in T.sections]
for sp in tpl_sect:
    for ref in sp.findall(qn("w:headerReference")) + sp.findall(qn("w:footerReference")):
        sp.remove(ref)


def plain(s):
    s = s.replace("\\&", "&").replace("---", "\u2014").replace("--", "\u2013").replace("``", "\u201c").replace("''", "\u201d")
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    return " ".join(s.replace("~", " ").split())


# --- remap body styles ---------------------------------------------------------------------------
def sid(p):
    ps = p._p.pPr.find(qn("w:pStyle")) if p._p.pPr is not None else None
    return ps.get(qn("w:val")) if ps is not None else ""


def no_list(p):
    """Turn off the style's automatic numbering on this paragraph (letters are written explicitly)."""
    pPr = p._p.get_or_add_pPr()
    num = pPr.makeelement(qn("w:numPr"), {})
    for tag, val in (("w:ilvl", "0"), ("w:numId", "0")):
        num.append(num.makeelement(qn(tag), {qn("w:val"): val}))
    pPr.insert(1, num)


first_h1, prev, sub = True, None, 0
for p in D.paragraphs:
    st, txt = sid(p), p.text
    if st in ("Heading1", "Heading2", "Heading3"):
        unnumbered = "QQUQQ" in txt
        for r in p.runs:
            r.text = r.text.replace("QQUQQ", "")
        if st == "Heading1":
            p.style = D.styles["H1"] if unnumbered else D.styles["H1_List (No Space)" if first_h1 else "H1_List (Space)"]
            first_h1 = first_h1 and unnumbered
            sub = 0
            for r in p.runs:
                r.text = r.text.upper()
        else:
            sub += 1
            p.style = D.styles["H2_First" if prev == "H1" else "H2_Cont"]
            no_list(p)
            p.runs[0].text = f"{chr(64 + sub)}. " + p.runs[0].text
            for r in p.runs:
                r.text = r.text.upper()
        prev = "H1" if st == "Heading1" else "H2"
        continue
    if "QQRQQ" in txt:
        for r in p.runs:
            r.text = r.text.replace("QQRQQ ", "").replace("QQRQQ", "")
        p.style = D.styles["References"]; prev = "ref"; continue
    if "QQEQ" in txt:
        for r in p.runs:
            if "QQEQ" in r.text:
                r.text = re.sub(r"\s*QQEQ(\d+)QQ", r"\t(\1)", r.text)
        p.style = D.styles["Equation"]; prev = "eq"
        pf = p.paragraph_format; pf.line_spacing_rule = docx.enum.text.WD_LINE_SPACING.SINGLE
        pf.space_before = Pt(6); pf.space_after = Pt(6); pf.left_indent = Inches(0); pf.first_line_indent = Inches(0)
        pf.tab_stops.add_tab_stop(Inches(3.183333), docx.enum.text.WD_TAB_ALIGNMENT.CLEAR)
        pf.tab_stops.add_tab_stop(Inches(3.25), docx.enum.text.WD_TAB_ALIGNMENT.RIGHT)
        continue
    if st == "ImageCaption":
        p.style = D.styles["Fig Caption"]; prev = "cap"; continue
    if st == "TableCaption":
        p.style = D.styles["Table Title"]; prev = "cap"; continue
    if st == "CaptionedFigure":
        p.alignment = 1; prev = "fig"; continue
    if st in ("FirstParagraph", "BodyText", "Compact", ""):
        p.style = D.styles["PARA_Indent" if prev == "para" and st == "BodyText" else "PARA"]
    prev = "para"

# pandoc puts each display equation in its own paragraph, before the paragraph holding its number: merge them
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
for p in list(D.paragraphs):
    if p.style.name == "Equation" and p._p.find(".//" + M + "oMath") is None:
        prv = p._p.getprevious()
        para = prv.find(".//" + M + "oMathPara") if prv is not None else None
        if para is not None:
            p._p.find(qn("w:r")).addprevious(para.find(M + "oMath"))
            prv.getparent().remove(prv)
# math: equation number on the equation's line; no clipping of tall inline math under exact line spacing
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
for p in D.paragraphs:
    if p._p.find(".//" + M + "oMath") is None:
        continue
    if p.style.name != "Equation":
        p.paragraph_format.line_spacing_rule = docx.enum.text.WD_LINE_SPACING.AT_LEAST
        p.paragraph_format.line_spacing = Pt(12)

# table captions go above the table, with "TABLE I" on its own line as in the template
tno = 0
for t in D.tables:
    tno += 1
    nxt, prv = t._tbl.getnext(), t._tbl.getprevious()
    cap = next((e for e in (prv, nxt) if e is not None and e.tag == qn("w:p") and docx.text.paragraph.Paragraph(e, D._body).style.name == "Table Title"), None)
    if cap is not None and cap is nxt:
        t._tbl.addprevious(cap)
    if cap is not None:
        lab = docx.text.paragraph.Paragraph(cap, D._body).insert_paragraph_before(f"TABLE {ROMAN[tno - 1]}", style=D.styles["Table Title"])
        lab.alignment = 1; docx.text.paragraph.Paragraph(cap, D._body).alignment = 1
# tables: one column wide, fixed layout, thin grid, 7.5 pt
COLW = 4810
for t in D.tables:
    tbl = t._tbl; tblPr = tbl.tblPr
    for tag in ("w:tblBorders", "w:tblLayout"):
        for e in tblPr.findall(qn(tag)):
            tblPr.remove(e)
    tblW = tblPr.find(qn("w:tblW")); tblW.set(qn("w:type"), "dxa"); tblW.set(qn("w:w"), str(COLW))
    borders = tblPr.makeelement(qn("w:tblBorders"), {})
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(borders.makeelement(qn(f"w:{edge}"), {qn("w:val"): "single", qn("w:sz"): "4", qn("w:space"): "0", qn("w:color"): "000000"}))
    layout = tblPr.makeelement(qn("w:tblLayout"), {qn("w:type"): "fixed"})
    tblW.addnext(borders); borders.addnext(layout)
    cols = tbl.tblGrid.findall(qn("w:gridCol")); tot = sum(int(c.get(qn("w:w"))) for c in cols)
    widths = []
    for c in cols:
        w = int(int(c.get(qn("w:w"))) * COLW / tot); c.set(qn("w:w"), str(w)); widths.append(w)
    for row in t.rows:
        for k, c in enumerate(row.cells):
            for cp in c.paragraphs:
                cp.style = D.styles["Normal"]; cp.paragraph_format.space_before = Pt(0); cp.paragraph_format.space_after = Pt(0)
                for r in cp.runs:
                    r.font.size = Pt(7.5)
# images: one column wide
for shp in D.inline_shapes:
    ratio = shp.height / shp.width
    shp.width = Inches(3.34); shp.height = int(Inches(3.34) * ratio)

# --- front matter ----------------------------------------------------------------------------
first = D.paragraphs[0]


def before(text, style, bold_label=None, sup=None):
    p = first.insert_paragraph_before("", style=D.styles[style])
    if bold_label:
        r = p.add_run(bold_label + " "); r.bold = True
    p.add_run(text)
    return p


before("Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.", "DOP")
before("Digital Object Identifier 10.1109/ACCESS.2024.Doi Number", "DOI")
before(plain(title), "Paper Title")
before("SANDANAYAKAGE JANINDU INDIRA SANDANAYAKA1, MANINDA EDIRISOORIYA1, ACHALA APONSO1, AND ADHISHA GAMMANPILA1", "AU")
before("1Informatics Institute of Technology, Colombo, Sri Lanka, in collaboration with the University of Westminster, London, U.K.", "PI")
before("Corresponding author: S. J. I. Sandanayaka (e-mail: janindu.20232270@iit.ac.lk).", "PI")
before("This work did not receive any specific funding.", "footnote text")
before(plain(abstract), "Abstract", bold_label="ABSTRACT")
last_front = before(plain(keywords), "IT", bold_label="INDEX TERMS")
# superscript affiliation markers
au = D.paragraphs[3]; au.clear()
for txt, sup in (("SANDANAYAKAGE JANINDU INDIRA SANDANAYAKA", False), ("1", True), (", MANINDA EDIRISOORIYA", False), ("1", True), (", ACHALA APONSO", False), ("1", True), (", AND ADHISHA GAMMANPILA", False), ("1", True)):
    r = au.add_run(txt); r.font.superscript = sup
pi = D.paragraphs[4]; pi.clear()
r = pi.add_run("1"); r.font.superscript = True
pi.add_run("Informatics Institute of Technology, Colombo, Sri Lanka, in collaboration with the University of Westminster, London, U.K. (e-mail: janindu.20232270@iit.ac.lk)")

# section 1: single column front matter; rest: continuous two columns (template's own section settings)
pPr = last_front._p.get_or_add_pPr()
pPr.append(copy.deepcopy(tpl_sect[0]))
body_el = D.element.body
old = body_el.find(qn("w:sectPr"))
body_el.replace(old, copy.deepcopy(tpl_sect[1]))

# author biographies (after the references), in the template's AU_Bios style with the name in bold capitals
for name, text in re.findall(r"\\begin\{IEEEbiography(?:nophoto)?\}(?:\[[^\]]*\])?\{([^}]*)\}(.*?)\\end\{IEEEbiography(?:nophoto)?\}", tex, re.S):
    bp = D.add_paragraph(style=D.styles["AU_Bios"])
    bp.add_run(plain(name).upper() + " ").bold = True
    bp.add_run(" ".join(plain(text).split()))

D.core_properties.title = plain(title)
D.save(str(HERE / "main.docx"))
print("wrote", HERE / "main.docx")
