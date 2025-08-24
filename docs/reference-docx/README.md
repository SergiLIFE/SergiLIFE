# Branded reference.docx for Word exports (SergiLIFE)

This guide creates a branded `docs/reference.docx` that Pandoc uses to style and watermark all Word exports in CI.

The CI workflow already looks for `docs/reference.docx` and uses it automatically when present.

## Brand defaults
- Brand: SergiLIFE
- Colors: Primary #0078D4, Secondary #00A1B3, Neutral text #2B2B2B, Accents #D9E7F6
- Watermark: "Confidential — Proprietary" (light gray, diagonal)
- Fonts: Headings Segoe UI Semibold (fallback Calibri, Arial), Body Segoe UI (fallback Calibri, Arial), Mono Consolas

## Create docs/reference.docx (once)
1) Open Word → New blank document.
2) Page and spacing
   - Page size A4; margins 1 inch all sides.
   - Line spacing 1.15; Paragraph spacing 6 pt after.
3) Theme colors (Design → Colors → Customize)
   - Accent 1 #0078D4, Accent 2 #00A1B3, Hyperlink #0078D4, Followed #5B9BD5.
4) Theme fonts (Design → Fonts → Customize)
   - Headings: Segoe UI Semibold; Body: Segoe UI. If Segoe UI is unavailable, use Calibri.
5) Define styles (Home → Styles → right-click → Modify)
   - Title: Segoe UI Semibold 28 pt, color #2B2B2B, centered, 12 pt before/after.
   - Subtitle: Segoe UI 14 pt, color #0078D4, centered, 6 pt before/after.
   - Heading 1: Segoe UI Semibold 18 pt, color #2B2B2B, Keep with next, optional 1 pt bottom border #D9E7F6.
   - Heading 2: Segoe UI Semibold 14 pt, color #2B2B2B, Keep with next.
   - Heading 3: Segoe UI 12 pt, color #2B2B2B.
   - Normal: Segoe UI 11 pt, color #2B2B2B.
   - Caption: Segoe UI 10 pt, italic, color #5A5A5A.
   - Code: Consolas 10 pt; optional light gray background; 0.5 cm left indent.
   - Quote: Italic, 2.25 pt left border #D9E7F6, 0.5 cm left indent.
   - Table Text: Segoe UI 10.5 pt.
   - List Bullet / Number: cohesive bullet/number styles; bullet color #2B2B2B.
6) Header & Footer (Insert → Header/Footer)
   - Header left: SergiLIFE (bold, Segoe UI Semibold 10 pt, #2B2B2B).
   - Header right: L.I.F.E THEORY (Segoe UI 10 pt).
   - Optional thin line under header: 0.5 pt #D9E7F6.
   - Footer left: © 2025 SergiLIFE. All rights reserved. Unauthorized commercial use prohibited.
   - Footer center: Page number (Bottom Center).
   - Footer right: Version placeholder "v1.0".
   - Enable Different first page if you want a cleaner cover header; keep footer for compliance.
7) Watermark (Design → Watermark → Custom)
   - Text: Confidential — Proprietary; Font: Segoe UI Light (or Calibri Light); Size: Auto; Color: Light Gray; Layout: Diagonal; Apply to Whole document.
   - NDA variant: use "Confidential — NDA Restricted".
8) Table design
   - Create a custom table style "SergiLIFE Table": header row background #D9E7F6, bold header text, thin borders #D9E7F6, alternating row shading ~3%, cell padding 0.12–0.18 in. Set as default table style.
9) Save as `docs/reference.docx` in the repo.

## Optional: Title page layout (content block)
You can let Pandoc generate a styled title using the reference styles, or you can maintain the cover in the template. If using Markdown/HTML blocks, Word may ignore some inline CSS; prefer mapping headings/paragraphs to styled elements defined above.

Suggested cover content (for manual insertion in Word or via styled Markdown):
- Title (Title style): L.I.F.E THEORY: A Self‑Evolving Neural Architecture for Autonomous Intelligence
- Subtitle (Subtitle style): Autonomous Learning, Clinical-Grade Latency, and Evidence-Based Optimization
- Accent rule below subtitle: 2 pt, #0078D4
- Metadata (centered Normal): SergiLIFE · v1.0 · 2025-08-22
- Confidential banner (text box): light border #D9E7F6; text: Confidential — Proprietary…

## CI usage (already wired)
- The docs workflow checks for `docs/reference.docx`:
  - If present: uses it for DOCX builds (watermark, styles, header/footer).
  - If missing: builds DOCX without reference (no watermark), with footer metadata only.
- PDF uses LaTeX watermark via `docs/whitepaper/pandoc.yaml` (no action needed).

## QA and tips
- Test locally:
  - `pandoc README.md --reference-doc=docs/reference.docx -o test.docx`
- Lock brand colors/fonts via Design → Colors/Fonts → Create New Theme Colors/Fonts.
- Maintain version in footer right or title page; you can pass `--metadata version="v1.0"` if you wire versioning.
- Keep an NDA variant at `docs/reference-nda.docx` and point Pandoc to it for restricted distributions.
