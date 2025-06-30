#!/usr/bin/env python3
"""
Visual Similarity Analysis using Screenshots and Image Embeddings

This script:
1. Renders HTML files using Playwright
2. Captures screenshots
3. Uses image embeddings (via OpenAI's CLIP or similar)
4. Analyzes visual similarity between pages
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.metrics.pairwise import cosine_similarity

# We'll use Playwright for rendering
try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing playwright...")
    os.system("pip install playwright")
    from playwright.async_api import async_playwright

# For image embeddings, we'll use CLIP
try:
    import torch
    import clip
    from PIL import Image
except ImportError:
    print("Installing required packages for image embeddings...")
    os.system("pip install torch torchvision pillow ftfy regex tqdm")
    os.system("pip install git+https://github.com/openai/CLIP.git")
    import torch
    import clip
    from PIL import Image


class VisualSimilarityAnalyzer:
    """Analyzes HTML files based on visual appearance using screenshots and image embeddings."""
    
    def __init__(self, html_dir: str = "sample_html", screenshots_dir: str = "screenshots"):
        self.html_dir = Path(html_dir)
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)
        
        self.html_files = {}
        self.screenshots = {}
        self.embeddings = {}
        self.similarity_matrix = None
        
        # Load CLIP model
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
    def load_html_files(self):
        """Load all HTML files from the directory."""
        print(f"\n📁 Loading HTML files from {self.html_dir}...")
        
        for file_path in self.html_dir.glob("*.html"):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.html_files[file_path.name] = file_path
                
        print(f"✓ Loaded {len(self.html_files)} HTML files")
        
    async def capture_screenshots(self):
        """Render HTML files and capture screenshots."""
        print("\n📸 Capturing screenshots...")
        
        async with async_playwright() as p:
            # Launch browser in headless mode
            browser = await p.chromium.launch(headless=True)
            
            # Set viewport size for consistency
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 1024},
                device_scale_factor=2  # Higher quality screenshots
            )
            
            page = await context.new_page()
            
            for filename, filepath in self.html_files.items():
                screenshot_path = self.screenshots_dir / f"{filename}.png"
                
                try:
                    # Navigate to the HTML file
                    await page.goto(f"file://{filepath.absolute()}")
                    
                    # Wait for page to load
                    await page.wait_for_load_state('networkidle')
                    
                    # Additional wait for dynamic content
                    await page.wait_for_timeout(2000)
                    
                    # Capture full page screenshot
                    await page.screenshot(
                        path=str(screenshot_path),
                        full_page=False  # Just viewport for consistency
                    )
                    
                    self.screenshots[filename] = screenshot_path
                    print(f"  ✓ Captured {filename}")
                    
                except Exception as e:
                    print(f"  ✗ Error capturing {filename}: {e}")
                    
            await browser.close()
            
    def generate_image_embeddings(self):
        """Generate CLIP embeddings for each screenshot."""
        print("\n🎨 Generating visual embeddings...")
        
        for filename, screenshot_path in self.screenshots.items():
            try:
                # Load and preprocess image
                image = Image.open(screenshot_path)
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # Generate embedding
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    # Normalize the features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                self.embeddings[filename] = image_features.cpu().numpy().flatten()
                print(f"  ✓ Embedded {filename}")
                
            except Exception as e:
                print(f"  ✗ Error embedding {filename}: {e}")
                
    def calculate_visual_similarity(self):
        """Calculate pairwise visual similarity between screenshots."""
        print("\n🎯 Calculating visual similarity...")
        
        filenames = list(self.embeddings.keys())
        n = len(filenames)
        
        # Create embedding matrix
        embedding_matrix = np.array([self.embeddings[f] for f in filenames])
        
        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(embedding_matrix)
        
        print(f"✓ Calculated similarity matrix ({n}x{n})")
        
    def visualize_results(self):
        """Create visualizations of visual similarity."""
        print("\n📊 Creating visualizations...")
        
        filenames = list(self.embeddings.keys())
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Similarity heatmap
        ax1 = plt.subplot(2, 3, 1)
        sns.heatmap(
            self.similarity_matrix,
            xticklabels=[f.replace('.html', '') for f in filenames],
            yticklabels=[f.replace('.html', '') for f in filenames],
            cmap='viridis',
            annot=True,
            fmt='.2f',
            ax=ax1
        )
        ax1.set_title('Visual Similarity Matrix (CLIP)', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. t-SNE visualization
        ax2 = plt.subplot(2, 3, 2)
        
        # Use t-SNE for 2D projection
        embeddings_matrix = np.array([self.embeddings[f] for f in filenames])
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(filenames)-1))
        coords = tsne.fit_transform(embeddings_matrix)
        
        # Color by filename pattern (PDPs vs others)
        colors = []
        for f in filenames:
            if 'stan' in f.lower() and 'pdp' not in f.lower():
                colors.append('#FF6B6B')  # Red for Stan products
            elif 'pdp' in f.lower():
                colors.append('#4ECDC4')  # Teal for PDPs
            elif 'cart' in f.lower():
                colors.append('#45B7D1')  # Blue for cart
            elif 'checkout' in f.lower():
                colors.append('#96CEB4')  # Green for checkout
            else:
                colors.append('#FFD93D')  # Yellow for others
                
        scatter = ax2.scatter(coords[:, 0], coords[:, 1], c=colors, s=200, alpha=0.7)
        
        # Add labels
        for i, filename in enumerate(filenames):
            ax2.annotate(
                filename.replace('.html', ''),
                (coords[i, 0], coords[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
            
        ax2.set_title('Visual Embedding Space (t-SNE)', fontsize=14)
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        
        # 3. MDS visualization
        ax3 = plt.subplot(2, 3, 3)
        
        # Use MDS for alternative 2D projection
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        distance_matrix = 1 - self.similarity_matrix
        mds_coords = mds.fit_transform(distance_matrix)
        
        scatter = ax3.scatter(mds_coords[:, 0], mds_coords[:, 1], c=colors, s=200, alpha=0.7)
        
        # Add labels
        for i, filename in enumerate(filenames):
            ax3.annotate(
                filename.replace('.html', ''),
                (mds_coords[i, 0], mds_coords[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
            
        ax3.set_title('Visual Embedding Space (MDS)', fontsize=14)
        ax3.set_xlabel('MDS Dimension 1')
        ax3.set_ylabel('MDS Dimension 2')
        
        # 4-6. Show sample screenshots
        for idx, (i, filename) in enumerate(list(self.screenshots.items())[:3]):
            ax = plt.subplot(2, 3, idx + 4)
            
            if filename in self.screenshots:
                img = Image.open(self.screenshots[filename])
                # Resize for display
                img.thumbnail((400, 300))
                ax.imshow(img)
                ax.set_title(f'{filename.replace(".html", "")}', fontsize=10)
                ax.axis('off')
        
        plt.suptitle('Visual Similarity Analysis using CLIP Embeddings', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visual_similarity_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to visual_similarity_analysis.png")
        
    def analyze_specific_pages(self):
        """Analyze specific page comparisons."""
        print("\n🔍 Visual Similarity Analysis:")
        
        # Find Stan Smith pages
        stan_pages = [f for f in self.embeddings.keys() if 'stan' in f.lower()]
        
        if len(stan_pages) >= 2:
            print(f"\nFound {len(stan_pages)} Stan-related pages")
            
            # Check if we have the specific pages
            if 'stans_white.html' in self.embeddings and 'stans_tan.html' in self.embeddings:
                filenames = list(self.embeddings.keys())
                white_idx = filenames.index('stans_white.html')
                tan_idx = filenames.index('stans_tan.html')
                
                visual_sim = self.similarity_matrix[white_idx, tan_idx]
                print(f"\nVisual similarity between Stan Smith variants:")
                print(f"  stans_white.html <-> stans_tan.html: {visual_sim:.3f}")
                print(f"  Interpretation: {'Very similar' if visual_sim > 0.8 else 'Moderately similar' if visual_sim > 0.6 else 'Different'} visual appearance")
                
    def save_results(self):
        """Save analysis results."""
        results = {
            'files_analyzed': list(self.embeddings.keys()),
            'similarity_matrix': self.similarity_matrix.tolist() if self.similarity_matrix is not None else None,
            'summary': {
                'total_files': len(self.embeddings),
                'average_similarity': float(np.mean(self.similarity_matrix)) if self.similarity_matrix is not None else 0,
                'embedding_dimension': len(next(iter(self.embeddings.values()))) if self.embeddings else 0
            }
        }
        
        with open('visual_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print("\n✓ Saved analysis results to visual_analysis_results.json")
        
    async def run_analysis(self):
        """Run the complete visual analysis pipeline."""
        self.load_html_files()
        await self.capture_screenshots()
        self.generate_image_embeddings()
        self.calculate_visual_similarity()
        self.visualize_results()
        self.analyze_specific_pages()
        self.save_results()


async def main():
    """Main execution function."""
    print("=" * 60)
    print("VISUAL SIMILARITY ANALYSIS")
    print("=" * 60)
    
    # First, make sure Playwright browsers are installed
    print("\n🌐 Ensuring Playwright browsers are installed...")
    os.system("playwright install chromium")
    
    analyzer = VisualSimilarityAnalyzer()
    await analyzer.run_analysis()
    
    print("\n✅ Visual analysis complete!")
    print("\nKey insights:")
    print("- Screenshots capture the visual appearance of pages")
    print("- CLIP embeddings encode visual features")
    print("- Visual similarity reflects design and layout patterns")
    print("- Product pages with same template should show high visual similarity")


if __name__ == "__main__":
    asyncio.run(main()) 