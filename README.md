# HTML Similarity Toolkit

A multi-dimensional toolkit for analysing and mapping websites.  It compares HTML pages through three complementary lenses:

1. **Semantic Similarity** – What does the page *say*?  Uses Voyage AI text embeddings + UMAP projection.  ( `embed_and_visualize.py` )
2. **Structural Similarity** – How is the page *built*?  Analyses DOM/ CSS templates, forms, link patterns, etc.  ( `structural_similarity.py` )
3. **Visual Similarity** – What does the page *look like*?  Renders screenshots with Playwright and embeds them with CLIP.  ( `visual_similarity.py` )

A fourth script (`compare_all_approaches.py`) stitches the outputs together to illustrate how the same pair of pages can appear close or far apart depending on the perspective.

---

## Why use it?

• **Content teams** – detect duplicate/related pages, power smarter recommendations, or audit SEO content.

• **Scraping / migration** – automatically classify page templates before a site move.

• **UX / QA** – catch inconsistent visual patterns or regressions across large sites.

---

## Quick-Start

```bash
# 1 . Install deps
pip install -r requirements.txt

# Playwright browsers (needed for the visual pipeline)
playwright install chromium

# 2 . Set your Voyage AI key  📜🔑
export VOYAGE_API_KEY="sk-..."

# 3 . Run one or all pipelines
python embed_and_visualize.py          # semantic
python structural_similarity.py        # structural
python visual_similarity.py            # visual (async, takes a bit longer)
python compare_all_approaches.py       # pretty comparison figure
```

Large artefacts (HTML, PNG, JSON) are stored with **Git LFS**.  Make sure you have it:

```bash
git lfs install   # one-time per machine
```

---

## Output files

| Script | Key output |
| ------ | ---------- |
| `embed_and_visualize.py` | `html_embeddings_visualization.png`, `storage.json` |
| `structural_similarity.py` | `structural_similarity_analysis.png`, `structural_analysis_results.json` |
| `visual_similarity.py` | `visual_similarity_analysis.png`, `visual_analysis_results.json`, screenshots/ |
| `compare_all_approaches.py` | `three_approaches_comparison.png` |

All plots are high-resolution and suitable for reports.

## Example: Combined Comparison
![Three Approaches to HTML Similarity: Semantic vs Structural vs Visual](three_approaches_comparison.png)

---

## Repository layout

```
.
├── .gitattributes              # Git LFS configuration for large files
├── README.md                   # Project overview and instructions
├── requirements.txt            # Python dependencies
├── embed_and_visualize.py      # Semantic similarity pipeline
├── structural_similarity.py    # Structural similarity pipeline
├── visual_similarity.py        # Visual similarity pipeline
├── compare_all_approaches.py   # Combined comparison visualization
├── combined_similarity.py      # Additional semantic helper script
├── compare_approaches.py       # Secondary comparison script
├── embed_with_rate_limits.py   # Embedding with rate-limit handling
├── incremental_embed.py        # Incremental embedding utility
├── test.py                     # Test and validation scripts
├── stub_module.bundle.js       # Bundled JS stub module
├── sample_html/                # Sample HTML input files (Adidas example)
│   ├── about.html
│   ├── blog.html
│   ├── contact.html
│   ├── homepage.html
│   ├── products.html
│   ├── stan_pdp.html
│   ├── stans.html
│   ├── stans_men.html
│   ├── stans_tan.html
│   └── stans_white.html
├── screenshots/                # Screenshots generated by visual pipeline (tracked via LFS)
│   ├── added_to_bag.html.png
│   ├── cart.html.png
│   ├── checkout.html.png
│   ├── home.html.png
│   ├── stan_pdp.html.png
│   ├── stans.html.png
│   ├── stans_men.html.png
│   ├── stans_tan.html.png
│   └── stans_white.html.png
├── *.png                       # Visualization outputs (tracked via LFS)
│   ├── html_embeddings_visualization.png
│   ├── combined_similarity_analysis.png
│   ├── semantic_vs_structural_comparison.png
│   ├── structural_similarity_analysis.png
│   ├── three_approaches_comparison.png
│   └── visual_similarity_analysis.png
├── *.json                      # Analysis result files (tracked via LFS)
│   ├── storage.json
│   ├── structural_analysis_results.json
│   └── visual_analysis_results.json
└── semantic_mesh.html          # Example semantic mesh HTML output
```

---

## Extending

• **Bring your own pages** – drop `.html` files into `sample_html/`.

• **Swap embedding models** – change the `model=` param in the semantic script; CLIP variant for visuals, etc.

• **Plug-in extra signals** – e.g. performance metrics, accessibility scores.

Contributions & suggestions welcome – feel free to open an issue or PR!  

---

## Licence
MIT © Lyman Labs 