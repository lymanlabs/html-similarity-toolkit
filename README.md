# HTML Embedding and Visualization Script

This script reads HTML files from a local directory, creates embeddings using Voyage AI, and visualizes them in 2D space using UMAP dimensionality reduction.

## Features

- 📁 Reads all HTML files from a specified directory
- 🧹 Extracts and cleans text content from HTML (removes scripts, styles, etc.)
- 🚀 Creates embeddings using Voyage AI models (default: voyage-code-2)
- 📊 Reduces dimensions using UMAP for 2D visualization
- 🎨 Creates beautiful scatter plot visualizations with file labels
- 📦 Handles batching for large numbers of files
- 🛡️ Robust error handling and logging

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Voyage AI API Key

You need a Voyage AI API key. Set it as an environment variable:

```bash
export VOYAGE_API_KEY="your_voyage_api_key_here"
```

Or you can modify the script to include your API key directly (not recommended for production).

### 3. Prepare HTML Files

Either:
- Create a `sample_html/` directory and add your HTML files
- Or run the script and it will create sample HTML files for you automatically

## Usage

### Basic Usage

```bash
python embed_and_visualize.py
```

This will:
1. Look for HTML files in the `sample_html/` directory
2. Create sample files if the directory doesn't exist
3. Extract text content from all HTML files
4. Create embeddings using voyage-code-2
5. Reduce to 2D using UMAP
6. Display and save a visualization

### Customization

You can modify the script to:

- **Change the embedding model**: Edit the `model` parameter in the `HTMLEmbeddingVisualizer` initialization
  ```python
  visualizer = HTMLEmbeddingVisualizer(model="voyage-3")  # or voyage-code-3, etc.
  ```

- **Change input directory**: Modify the directory parameter
  ```python
  visualizer.run(directory="my_html_files")
  ```

- **Adjust UMAP parameters**: Modify the `reduce_dimensions` method for different clustering behavior

## Supported Models

The script supports all Voyage AI embedding models:
- `voyage-code-2` (default) - Optimized for code-related content
- `voyage-3` - General purpose, latest generation
- `voyage-3-lite` - Faster, lower cost
- `voyage-code-3` - Latest code-optimized model
- `voyage-finance-2` - Finance domain-specific
- `voyage-law-2` - Legal domain-specific

## Output

The script will:
1. Display detailed logging information during processing
2. Show an interactive matplotlib plot
3. Save a high-resolution PNG file: `html_embeddings_visualization.png`

## Example Output

The visualization will show:
- Each HTML file as a colored point in 2D space
- File names labeled next to each point
- Files with similar content clustered together
- A statistics box showing file count and model information

## File Structure

```
.
├── embed_and_visualize.py      # Main script
├── requirements.txt            # Dependencies
├── sample_html/               # Directory for HTML files
│   ├── homepage.html          # (auto-generated if missing)
│   ├── about.html
│   ├── contact.html
│   ├── products.html
│   └── blog.html
└── html_embeddings_visualization.png  # Output visualization
```

## Troubleshooting

### API Key Issues
- Ensure `VOYAGE_API_KEY` is set correctly
- Check that your API key has sufficient credits
- Verify you're using a valid Voyage AI API key

### Memory Issues
- For large numbers of files, the script automatically batches requests
- If you run out of memory, try reducing the batch size in the `create_embeddings` method

### No HTML Files Found
- Check that your HTML files have `.html` or `.htm` extensions
- Ensure the directory path is correct
- The script will create sample files if none are found

### Visualization Issues
- Make sure you have a display available (for headless servers, consider saving only)
- Install additional backends if matplotlib has display issues

## Cost Estimation

With voyage-code-2 pricing ($0.12 per million tokens):
- Small HTML files (~500 tokens each): ~100 files = $0.006
- Medium HTML files (~2000 tokens each): ~100 files = $0.024
- Large HTML files (~5000 tokens each): ~100 files = $0.060

Remember: First 50 million tokens are free for voyage-code-2! 