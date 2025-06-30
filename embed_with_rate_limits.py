#!/usr/bin/env python3
"""
HTML Embedding and Visualization Script with Rate Limiting

This version handles Voyage AI rate limits by adding delays between API calls.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict
import re

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import umap
import voyageai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HTMLEmbeddingVisualizerRateLimited:
    """Class to handle HTML file embedding and visualization with rate limiting."""
    
    def __init__(self, api_key: str = None, model: str = "voyage-code-2"):
        """
        Initialize the visualizer.
        
        Args:
            api_key: Voyage AI API key. If None, will try to get from environment.
            model: Voyage AI model to use for embeddings.
        """
        self.model = model
        
        # Initialize Voyage AI client
        try:
            if api_key:
                self.voyage_client = voyageai.Client(api_key=api_key)
            else:
                self.voyage_client = voyageai.Client()  # Uses VOYAGE_API_KEY env var
            logger.info(f"Initialized Voyage AI client with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize Voyage AI client: {e}")
            sys.exit(1)
    
    def extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.warning(f"Error extracting text from HTML: {e}")
            return html_content  # Return raw content as fallback
    
    def read_html_files(self, directory: str = "sample_html") -> List[Tuple[str, str]]:
        """Read all HTML files from the specified directory."""
        html_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory '{directory}' does not exist!")
            return []
        
        # Find all HTML files
        html_extensions = ['.html', '.htm']
        for ext in html_extensions:
            for file_path in directory_path.glob(f"*{ext}"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Extract text from HTML
                    text_content = self.extract_text_from_html(content)
                    
                    if text_content.strip():  # Only include files with content
                        html_files.append((file_path.name, text_content))
                        logger.info(f"Read file: {file_path.name}")
                    else:
                        logger.warning(f"Skipping empty file: {file_path.name}")
                        
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
        
        logger.info(f"Successfully read {len(html_files)} HTML files")
        return html_files
    
    def create_embeddings_with_rate_limit(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for the given texts with rate limiting.
        
        Args:
            texts: List of text content to embed
            
        Returns:
            NumPy array of embeddings
        """
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts using {self.model} with rate limiting...")
            
            all_embeddings = []
            
            # Process one file at a time to respect rate limits
            for i, text in enumerate(texts):
                logger.info(f"Processing file {i+1}/{len(texts)}...")
                
                try:
                    result = self.voyage_client.embed(
                        [text],  # Process one at a time
                        model=self.model,
                        input_type="document"
                    )
                    
                    all_embeddings.extend(result.embeddings)
                    logger.info(f"✓ Successfully embedded file {i+1}")
                    
                    # Add delay between requests to respect rate limits (3 RPM = 20 seconds between requests)
                    if i < len(texts) - 1:  # Don't wait after the last one
                        logger.info("Waiting 25 seconds for rate limit...")
                        time.sleep(25)
                        
                except Exception as e:
                    logger.error(f"Error embedding file {i+1}: {e}")
                    # Add a longer delay if we hit rate limits
                    if "rate limit" in str(e).lower():
                        logger.info("Rate limit hit, waiting 60 seconds...")
                        time.sleep(60)
                    else:
                        raise e
            
            embeddings_array = np.array(all_embeddings)
            logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            sys.exit(1)
    
    def reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce dimensionality using UMAP."""
        try:
            logger.info(f"Reducing dimensions from {embeddings.shape[1]} to {n_components} using UMAP...")
            
            # Configure UMAP
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(embeddings) - 1),  # Adjust for small datasets
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            
            reduced_embeddings = reducer.fit_transform(embeddings)
            logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")
            return reduced_embeddings
            
        except Exception as e:
            logger.error(f"Error reducing dimensions: {e}")
            sys.exit(1)
    
    def create_visualization(self, reduced_embeddings: np.ndarray, filenames: List[str], 
                           save_path: str = "html_embeddings_visualization.png"):
        """Create and save a 2D visualization of the embeddings."""
        try:
            logger.info("Creating visualization...")
            
            # Set up the plot style
            plt.style.use('default')
            sns.set_palette("husl")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create scatter plot
            scatter = ax.scatter(
                reduced_embeddings[:, 0], 
                reduced_embeddings[:, 1],
                s=100,
                alpha=0.7,
                c=range(len(filenames)),
                cmap='tab20'
            )
            
            # Add labels for each point
            for i, filename in enumerate(filenames):
                ax.annotate(
                    filename, 
                    (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    ha='left'
                )
            
            # Customize the plot
            ax.set_xlabel('UMAP Dimension 1', fontsize=12)
            ax.set_ylabel('UMAP Dimension 2', fontsize=12)
            ax.set_title(f'HTML Files Embedding Visualization\n({self.model} + UMAP)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add some statistics as text
            stats_text = f"Files: {len(filenames)}\nModel: {self.model}\nEmbedding Dim: {reduced_embeddings.shape[1]}D"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
            
            # Show the plot
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def run(self, directory: str = "sample_html"):
        """Run the complete pipeline with rate limiting."""
        logger.info("Starting HTML embedding and visualization pipeline with rate limiting...")
        
        # Step 1: Read HTML files
        html_files = self.read_html_files(directory)
        if not html_files:
            logger.error("No HTML files found to process!")
            return
        
        filenames = [filename for filename, _ in html_files]
        texts = [content for _, content in html_files]
        
        # Estimate time
        estimated_minutes = len(texts) * 25 / 60
        logger.info(f"⏱️  Estimated processing time: {estimated_minutes:.1f} minutes for {len(texts)} files")
        
        # Step 2: Create embeddings with rate limiting
        embeddings = self.create_embeddings_with_rate_limit(texts)
        
        # Step 3: Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(embeddings)
        
        # Step 4: Create visualization
        self.create_visualization(reduced_embeddings, filenames)
        
        logger.info("Pipeline completed successfully!")


def main():
    """Main function to run the embedding and visualization pipeline."""
    
    # Initialize and run the visualizer
    try:
        visualizer = HTMLEmbeddingVisualizerRateLimited(
            api_key=None,  # Will use VOYAGE_API_KEY environment variable
            model="voyage-code-2"
        )
        
        visualizer.run()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 