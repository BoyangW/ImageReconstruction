from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def resolve_image(md_path: Path, image_ref: str) -> Path:
    path = Path(image_ref)
    if path.is_absolute():
        return path
    return (md_path.parent / path).resolve()


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(7.5 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if all(part.replace("-", "").replace(":", "").strip() == "" for part in parts):
            continue
        rows.append(parts)
    return rows


def markdown_to_story(md_path: Path) -> list:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.grey))
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 14
    styles["Heading1"].fontSize = 20
    styles["Heading1"].leading = 24
    styles["Heading2"].fontSize = 14
    styles["Heading2"].leading = 18

    lines = md_path.read_text(encoding="utf-8").splitlines()
    story = []
    paragraph: list[str] = []
    table_buffer: list[str] = []
    code_buffer: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(paragraph)
            text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
            text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 0.08 * inch))
            paragraph = []

    def flush_table() -> None:
        nonlocal table_buffer
        if table_buffer:
            rows = parse_table(table_buffer)
            if rows:
                table = Table(rows, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 0), (-1, -1), 7),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.14 * inch))
            table_buffer = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                story.append(Preformatted("\n".join(code_buffer), styles["Code"]))
                story.append(Spacer(1, 0.1 * inch))
                code_buffer = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_buffer.append(line)
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_buffer.append(line)
            continue
        else:
            flush_table()

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            flush_paragraph()
            image_path = resolve_image(md_path, image_match.group(2))
            if image_path.exists():
                img = Image(str(image_path))
                max_width = 6.6 * inch
                max_height = 4.2 * inch
                scale = min(max_width / img.imageWidth, max_height / img.imageHeight, 1.0)
                img.drawWidth = img.imageWidth * scale
                img.drawHeight = img.imageHeight * scale
                story.append(img)
                caption = image_match.group(1)
                if caption:
                    story.append(Paragraph(caption, styles["Caption"]))
                story.append(Spacer(1, 0.16 * inch))
            continue

        if stripped == "":
            flush_paragraph()
            continue
        if stripped == "\\pagebreak":
            flush_paragraph()
            story.append(PageBreak())
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(stripped[2:], styles["Heading1"]))
            story.append(Spacer(1, 0.12 * inch))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(stripped[3:], styles["Heading2"]))
            story.append(Spacer(1, 0.08 * inch))
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            text = stripped[2:]
            text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
            story.append(Paragraph(text, styles["BodyText"], bulletText="•"))
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            story.append(Paragraph(f"<i>{stripped[2:]}</i>", styles["BodyText"]))
            story.append(Spacer(1, 0.08 * inch))
            continue

        paragraph.append(stripped)

    flush_paragraph()
    flush_table()
    return story


def render_pdf(md_path: Path, pdf_path: Path) -> None:
    story = markdown_to_story(md_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=md_path.stem,
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a simple Markdown file to PDF.")
    parser.add_argument("markdown", type=Path)
    parser.add_argument("pdf", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_pdf(args.markdown, args.pdf)


if __name__ == "__main__":
    main()

