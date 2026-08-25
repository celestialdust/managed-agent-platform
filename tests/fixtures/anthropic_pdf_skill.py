"""Anthropic's real, public `pdf` Agent Skill, vendored verbatim as a test fixture.

Here so the cluster tests can register the genuine article instead of a toy SKILL.md we
wrote ourselves. A hand-written fixture only ever exercises the shapes we already
thought of; this one carries a 437-character description, a third frontmatter key we do
not model, and instructions that point at sibling files -- all of which the platform
either handles or visibly does not.

Upstream
    https://github.com/anthropics/skills/tree/main/skills/pdf
    https://raw.githubusercontent.com/anthropics/skills/main/skills/pdf/SKILL.md
    Fetched 2026-08-23 from `main` at commit 1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563
    ("Update skill-creator and make scripts executable (#350)", 2026-02-06). The
    SKILL.md blob itself is d3e046a5ae107a6cb23cfb16c219837094ab35d3; the fetched
    bytes are 8072 long and hash to
    sha256:9f78b8359fbd4943ad260a7a1e436e5a96503406d6c34e99f69223d647d85b9c.

Licence
    Proprietary, as the document's own frontmatter says: "Proprietary. LICENSE.txt has
    complete terms". The directory's LICENSE.txt opens "(c) 2025 Anthropic, PBC. All
    rights reserved." and governs use by your agreement with Anthropic, or failing
    that Anthropic's Consumer or Commercial Terms. Not open source. Vendored here for
    testing our own delivery path against a real skill, nothing else -- and only the
    SKILL.md text, never the sibling scripts.

What our platform cannot satisfy today
    1.  The siblings are not delivered. `POST /v1/skills` takes `SkillUpload`, which
        has exactly one field, `skill_md`, under `extra="forbid"`. There is no field
        for a second file. So `REFERENCED_FILES` below arrive nowhere, and an agent
        that follows this body's "read FORMS.md" branch is sent to a file that does
        not exist in its workspace. The text stays verbatim rather than being edited
        to remove those pointers, because the dangling pointer is the finding.
    2.  The names in the body do not match the names upstream. The body says
        `REFERENCE.md` and `FORMS.md`; the files in that directory are `reference.md`
        and `forms.md`. Anything that resolves them case-sensitively misses even once
        we can deliver them.
    3.  One recipe in it cannot run, and it is the OCR one. The Session image now
        installs all six of `REQUIRED_PYTHON_PACKAGES` and the two command-line
        programs this body actually shells out to, `pdftotext`/`pdfimages` from
        poppler-utils and `qpdf`. What is missing is `tesseract`, the binary
        `pytesseract` binds to: Amazon Linux 2023 does not package it under any name.
        The wrapper is installed anyway, so the OCR branch reaches a missing binary
        rather than a missing module.

        `pdftk` is also absent and is NOT a gap: this document's own heading reads
        "pdftk (if available)" and its summary table routes command-line merging to
        qpdf, which is present.

        This body's "pip install" lines cannot be followed, and that is why the six
        are baked into the image instead. Measured inside the sandbox on a live pod:
        there is no `pip` on the agent's PATH at all (`uv venv` seeds none, and
        nothing else provides one), `/opt/map/venv` is a read-only filesystem, and so
        is `/tmp` -- only `/session/workspace` is writable. So a runtime install would
        need a seeded pip, `--target /session/workspace/...`, and a `PYTHONPATH`, none
        of which this document mentions. The binaries are further out of reach still:
        `dnf` wants root and the agent is uid 10001. Pre-installing does not work
        around this document, it makes this document's setup step unnecessary.

What it does clear, checked rather than assumed
    8072 bytes is under `SKILL_MD_MAX_BYTES` (32 KiB); the description is 437
    characters against a `DESCRIPTION_MAX_CHARS` of 1024; `pdf` matches
    `SKILL_NAME_PATTERN`; and the frontmatter's third key, `license`, is ignored by
    `parse_skill_md` and preserved by `ValidatedSkill.text`. So this registers.
"""

from typing import Final

REQUIRED_PYTHON_PACKAGES: Final[tuple[str, ...]] = (
    "pandas",
    "pdf2image",
    "pdfplumber",
    "pypdf",
    "pytesseract",
    "reportlab",
)
"""Third-party Python distributions this SKILL.md tells the agent to import or install.

Read off the document, not guessed. The lines each name came from, verbatim:

    "from pypdf import PdfReader, PdfWriter"        -> pypdf
    "import pdfplumber"                             -> pdfplumber
    "import pandas as pd"                           -> pandas
    "from reportlab.lib.pagesizes import letter"    -> reportlab
    "# Requires: pip install pytesseract pdf2image" -> pytesseract, pdf2image
    "import pytesseract"                            -> pytesseract
    "from pdf2image import convert_from_path"       -> pdf2image

Three near-misses are deliberately out. `pypdfium2` appears once, as "For advanced
pypdfium2 usage, see REFERENCE.md" -- a pointer, not an import, and the import it
refers to is in a file we do not vendor. `openpyxl` is never named, yet
`combined_df.to_excel("extracted_tables.xlsx", index=False)` cannot run without it; it
is a real hidden requirement and belongs in the docstring rather than in a tuple that
claims to be derived from the text. `pdf-lib` is JavaScript.
"""

REFERENCED_FILES: Final[tuple[str, ...]] = (
    "forms.md",
    "reference.md",
)
"""Sibling files this SKILL.md sends the agent to read, by their real upstream paths.

Named here at the paths that exist upstream, lowercase, not at the uppercase spellings
the body uses -- see the docstring above. Neither reaches a Session: `skill_md` is the
whole request body. `forms.md` in turn drives eight scripts under `scripts/`
(check_bounding_boxes.py, check_fillable_fields.py, convert_pdf_to_images.py,
create_validation_image.py, extract_form_field_info.py, extract_form_structure.py,
fill_fillable_fields.py, fill_pdf_form_with_annotations.py), which are two hops from
anything we deliver. They are listed rather than vendored on purpose.
"""

SKILL_MD: Final[str] = (
    "---\n"
    "name: pdf\n"
    "description: Use this skill whenever the user wants to do anything with "
    "PDF files. This includes reading or extracting text/tables from PDFs, co"
    "mbining or merging multiple PDFs into one, splitting PDFs apart, rotatin"
    "g pages, adding watermarks, creating new PDFs, filling PDF forms, encryp"
    "ting/decrypting PDFs, extracting images, and OCR on scanned PDFs to make"
    " them searchable. If the user mentions a .pdf file or asks to produce on"
    "e, use this skill.\n"
    "license: Proprietary. LICENSE.txt has complete terms\n"
    "---\n"
    "\n"
    "# PDF Processing Guide\n"
    "\n"
    "## Overview\n"
    "\n"
    "This guide covers essential PDF processing operations using Python libra"
    "ries and command-line tools. For advanced features, JavaScript libraries"
    ", and detailed examples, see REFERENCE.md. If you need to fill out a PDF"
    " form, read FORMS.md and follow its instructions.\n"
    "\n"
    "## Quick Start\n"
    "\n"
    "```python\n"
    "from pypdf import PdfReader, PdfWriter\n"
    "\n"
    "# Read a PDF\n"
    'reader = PdfReader("document.pdf")\n'
    'print(f"Pages: {len(reader.pages)}")\n'
    "\n"
    "# Extract text\n"
    'text = ""\n'
    "for page in reader.pages:\n"
    "    text += page.extract_text()\n"
    "```\n"
    "\n"
    "## Python Libraries\n"
    "\n"
    "### pypdf - Basic Operations\n"
    "\n"
    "#### Merge PDFs\n"
    "```python\n"
    "from pypdf import PdfWriter, PdfReader\n"
    "\n"
    "writer = PdfWriter()\n"
    'for pdf_file in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:\n'
    "    reader = PdfReader(pdf_file)\n"
    "    for page in reader.pages:\n"
    "        writer.add_page(page)\n"
    "\n"
    'with open("merged.pdf", "wb") as output:\n'
    "    writer.write(output)\n"
    "```\n"
    "\n"
    "#### Split PDF\n"
    "```python\n"
    'reader = PdfReader("input.pdf")\n'
    "for i, page in enumerate(reader.pages):\n"
    "    writer = PdfWriter()\n"
    "    writer.add_page(page)\n"
    '    with open(f"page_{i+1}.pdf", "wb") as output:\n'
    "        writer.write(output)\n"
    "```\n"
    "\n"
    "#### Extract Metadata\n"
    "```python\n"
    'reader = PdfReader("document.pdf")\n'
    "meta = reader.metadata\n"
    'print(f"Title: {meta.title}")\n'
    'print(f"Author: {meta.author}")\n'
    'print(f"Subject: {meta.subject}")\n'
    'print(f"Creator: {meta.creator}")\n'
    "```\n"
    "\n"
    "#### Rotate Pages\n"
    "```python\n"
    'reader = PdfReader("input.pdf")\n'
    "writer = PdfWriter()\n"
    "\n"
    "page = reader.pages[0]\n"
    "page.rotate(90)  # Rotate 90 degrees clockwise\n"
    "writer.add_page(page)\n"
    "\n"
    'with open("rotated.pdf", "wb") as output:\n'
    "    writer.write(output)\n"
    "```\n"
    "\n"
    "### pdfplumber - Text and Table Extraction\n"
    "\n"
    "#### Extract Text with Layout\n"
    "```python\n"
    "import pdfplumber\n"
    "\n"
    'with pdfplumber.open("document.pdf") as pdf:\n'
    "    for page in pdf.pages:\n"
    "        text = page.extract_text()\n"
    "        print(text)\n"
    "```\n"
    "\n"
    "#### Extract Tables\n"
    "```python\n"
    'with pdfplumber.open("document.pdf") as pdf:\n'
    "    for i, page in enumerate(pdf.pages):\n"
    "        tables = page.extract_tables()\n"
    "        for j, table in enumerate(tables):\n"
    '            print(f"Table {j+1} on page {i+1}:")\n'
    "            for row in table:\n"
    "                print(row)\n"
    "```\n"
    "\n"
    "#### Advanced Table Extraction\n"
    "```python\n"
    "import pandas as pd\n"
    "\n"
    'with pdfplumber.open("document.pdf") as pdf:\n'
    "    all_tables = []\n"
    "    for page in pdf.pages:\n"
    "        tables = page.extract_tables()\n"
    "        for table in tables:\n"
    "            if table:  # Check if table is not empty\n"
    "                df = pd.DataFrame(table[1:], columns=table[0])\n"
    "                all_tables.append(df)\n"
    "\n"
    "# Combine all tables\n"
    "if all_tables:\n"
    "    combined_df = pd.concat(all_tables, ignore_index=True)\n"
    '    combined_df.to_excel("extracted_tables.xlsx", index=False)\n'
    "```\n"
    "\n"
    "### reportlab - Create PDFs\n"
    "\n"
    "#### Basic PDF Creation\n"
    "```python\n"
    "from reportlab.lib.pagesizes import letter\n"
    "from reportlab.pdfgen import canvas\n"
    "\n"
    'c = canvas.Canvas("hello.pdf", pagesize=letter)\n'
    "width, height = letter\n"
    "\n"
    "# Add text\n"
    'c.drawString(100, height - 100, "Hello World!")\n'
    'c.drawString(100, height - 120, "This is a PDF created with reportlab"'
    ")\n"
    "\n"
    "# Add a line\n"
    "c.line(100, height - 140, 400, height - 140)\n"
    "\n"
    "# Save\n"
    "c.save()\n"
    "```\n"
    "\n"
    "#### Create PDF with Multiple Pages\n"
    "```python\n"
    "from reportlab.lib.pagesizes import letter\n"
    "from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Pag"
    "eBreak\n"
    "from reportlab.lib.styles import getSampleStyleSheet\n"
    "\n"
    'doc = SimpleDocTemplate("report.pdf", pagesize=letter)\n'
    "styles = getSampleStyleSheet()\n"
    "story = []\n"
    "\n"
    "# Add content\n"
    "title = Paragraph(\"Report Title\", styles['Title'])\n"
    "story.append(title)\n"
    "story.append(Spacer(1, 12))\n"
    "\n"
    'body = Paragraph("This is the body of the report. " * 20, styles[\'Norm'
    "al'])\n"
    "story.append(body)\n"
    "story.append(PageBreak())\n"
    "\n"
    "# Page 2\n"
    "story.append(Paragraph(\"Page 2\", styles['Heading1']))\n"
    "story.append(Paragraph(\"Content for page 2\", styles['Normal']))\n"
    "\n"
    "# Build PDF\n"
    "doc.build(story)\n"
    "```\n"
    "\n"
    "#### Subscripts and Superscripts\n"
    "\n"
    "**IMPORTANT**: Never use Unicode subscript/superscript characters (₀₁₂₃₄"
    "₅₆₇₈₉, ⁰¹²³⁴⁵⁶⁷⁸⁹) in ReportLab PDFs. The built-in fonts do not include "
    "these glyphs, causing them to render as solid black boxes.\n"
    "\n"
    "Instead, use ReportLab's XML markup tags in Paragraph objects:\n"
    "```python\n"
    "from reportlab.platypus import Paragraph\n"
    "from reportlab.lib.styles import getSampleStyleSheet\n"
    "\n"
    "styles = getSampleStyleSheet()\n"
    "\n"
    "# Subscripts: use <sub> tag\n"
    "chemical = Paragraph(\"H<sub>2</sub>O\", styles['Normal'])\n"
    "\n"
    "# Superscripts: use <super> tag\n"
    'squared = Paragraph("x<super>2</super> + y<super>2</super>", styles[\'N'
    "ormal'])\n"
    "```\n"
    "\n"
    "For canvas-drawn text (not Paragraph objects), manually adjust font the "
    "size and position rather than using Unicode subscripts/superscripts.\n"
    "\n"
    "## Command-Line Tools\n"
    "\n"
    "### pdftotext (poppler-utils)\n"
    "```bash\n"
    "# Extract text\n"
    "pdftotext input.pdf output.txt\n"
    "\n"
    "# Extract text preserving layout\n"
    "pdftotext -layout input.pdf output.txt\n"
    "\n"
    "# Extract specific pages\n"
    "pdftotext -f 1 -l 5 input.pdf output.txt  # Pages 1-5\n"
    "```\n"
    "\n"
    "### qpdf\n"
    "```bash\n"
    "# Merge PDFs\n"
    "qpdf --empty --pages file1.pdf file2.pdf -- merged.pdf\n"
    "\n"
    "# Split pages\n"
    "qpdf input.pdf --pages . 1-5 -- pages1-5.pdf\n"
    "qpdf input.pdf --pages . 6-10 -- pages6-10.pdf\n"
    "\n"
    "# Rotate pages\n"
    "qpdf input.pdf output.pdf --rotate=+90:1  # Rotate page 1 by 90 degrees"
    "\n"
    "\n"
    "# Remove password\n"
    "qpdf --password=mypassword --decrypt encrypted.pdf decrypted.pdf\n"
    "```\n"
    "\n"
    "### pdftk (if available)\n"
    "```bash\n"
    "# Merge\n"
    "pdftk file1.pdf file2.pdf cat output merged.pdf\n"
    "\n"
    "# Split\n"
    "pdftk input.pdf burst\n"
    "\n"
    "# Rotate\n"
    "pdftk input.pdf rotate 1east output rotated.pdf\n"
    "```\n"
    "\n"
    "## Common Tasks\n"
    "\n"
    "### Extract Text from Scanned PDFs\n"
    "```python\n"
    "# Requires: pip install pytesseract pdf2image\n"
    "import pytesseract\n"
    "from pdf2image import convert_from_path\n"
    "\n"
    "# Convert PDF to images\n"
    "images = convert_from_path('scanned.pdf')\n"
    "\n"
    "# OCR each page\n"
    'text = ""\n'
    "for i, image in enumerate(images):\n"
    '    text += f"Page {i+1}:\\n"\n'
    "    text += pytesseract.image_to_string(image)\n"
    '    text += "\\n\\n"\n'
    "\n"
    "print(text)\n"
    "```\n"
    "\n"
    "### Add Watermark\n"
    "```python\n"
    "from pypdf import PdfReader, PdfWriter\n"
    "\n"
    "# Create watermark (or load existing)\n"
    'watermark = PdfReader("watermark.pdf").pages[0]\n'
    "\n"
    "# Apply to all pages\n"
    'reader = PdfReader("document.pdf")\n'
    "writer = PdfWriter()\n"
    "\n"
    "for page in reader.pages:\n"
    "    page.merge_page(watermark)\n"
    "    writer.add_page(page)\n"
    "\n"
    'with open("watermarked.pdf", "wb") as output:\n'
    "    writer.write(output)\n"
    "```\n"
    "\n"
    "### Extract Images\n"
    "```bash\n"
    "# Using pdfimages (poppler-utils)\n"
    "pdfimages -j input.pdf output_prefix\n"
    "\n"
    "# This extracts all images as output_prefix-000.jpg, output_prefix-001.j"
    "pg, etc.\n"
    "```\n"
    "\n"
    "### Password Protection\n"
    "```python\n"
    "from pypdf import PdfReader, PdfWriter\n"
    "\n"
    'reader = PdfReader("input.pdf")\n'
    "writer = PdfWriter()\n"
    "\n"
    "for page in reader.pages:\n"
    "    writer.add_page(page)\n"
    "\n"
    "# Add password\n"
    'writer.encrypt("userpassword", "ownerpassword")\n'
    "\n"
    'with open("encrypted.pdf", "wb") as output:\n'
    "    writer.write(output)\n"
    "```\n"
    "\n"
    "## Quick Reference\n"
    "\n"
    "| Task | Best Tool | Command/Code |\n"
    "|------|-----------|--------------|\n"
    "| Merge PDFs | pypdf | `writer.add_page(page)` |\n"
    "| Split PDFs | pypdf | One page per file |\n"
    "| Extract text | pdfplumber | `page.extract_text()` |\n"
    "| Extract tables | pdfplumber | `page.extract_tables()` |\n"
    "| Create PDFs | reportlab | Canvas or Platypus |\n"
    "| Command line merge | qpdf | `qpdf --empty --pages ...` |\n"
    "| OCR scanned PDFs | pytesseract | Convert to image first |\n"
    "| Fill PDF forms | pdf-lib or pypdf (see FORMS.md) | See FORMS.md |\n"
    "\n"
    "## Next Steps\n"
    "\n"
    "- For advanced pypdfium2 usage, see REFERENCE.md\n"
    "- For JavaScript libraries (pdf-lib), see REFERENCE.md\n"
    "- If you need to fill out a PDF form, follow the instructions in FORMS.m"
    "d\n"
    "- For troubleshooting guides, see REFERENCE.md\n"
)
"""The document exactly as fetched, byte for byte, newline-terminated.

Split across implicit-concatenation chunks only because this repo's source lines stop
at 88 columns and the description line upstream is 450 characters wide. The chunking
is a property of the source file and not of the value: concatenated, this is the same
8072 bytes named in the module docstring's hash, and any test may rely on that.

Verbatim is the point. A fixture reflowed to fit, or with its `REFERENCE.md` pointers
tidied away, would stop being evidence about what the platform does with a real skill.
"""
