#!/usr/bin/env python3
"""
Incremental HTML Embedding and Visualization Script

This version saves embeddings to disk and only processes new files.
"""

import os
import sys
import logging
import time
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import hashlib

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

class IncrementalHTMLEmbeddingVisualizer:
    """Class to handle incremental HTML file embedding and visualization."""
    
    def __init__(self, api_key: str = None, model: str = "voyage-code-2", 
                 cache_file: str = "embeddings_cache.pkl"):
        """
        Initialize the visualizer.
        
        Args:
            api_key: Voyage AI API key. If None, will try to get from environment.
            model: Voyage AI model to use for embeddings.
            cache_file: File to store embeddings cache.
        """
        self.model = model
        self.cache_file = cache_file
        
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
    
    def get_file_hash(self, content: str) -> str:
        """Get hash of file content for change detection."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_cache(self) -> Dict:
        """Load existing embeddings cache."""
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded cache with {len(cache.get('embeddings', {}))} existing embeddings")
                return cache
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
                return {"embeddings": {}, "filenames": [], "content_hashes": {}}
        else:
            logger.info("No existing cache found, starting fresh")
            return {"embeddings": {}, "filenames": [], "content_hashes": {}}
    
    def save_cache(self, cache: Dict):
        """Save embeddings cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache, f)
            logger.info(f"Saved cache with {len(cache['embeddings'])} embeddings")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
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
    
    def read_html_files(self, directory: str = "sample_html") -> List[Tuple[str, str, str]]:
        """Read all HTML files and return (filename, content, hash)."""
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
                        raw_content = f.read()
                        
                    # Extract text from HTML
                    text_content = self.extract_text_from_html(raw_content)
                    
                    if text_content.strip():  # Only include files with content
                        content_hash = self.get_file_hash(text_content)
                        html_files.append((file_path.name, text_content, content_hash))
                        logger.info(f"Read file: {file_path.name}")
                    else:
                        logger.warning(f"Skipping empty file: {file_path.name}")
                        
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
        
        logger.info(f"Successfully read {len(html_files)} HTML files")
        return html_files
    
    def identify_new_files(self, html_files: List[Tuple[str, str, str]], 
                          cache: Dict) -> List[Tuple[str, str]]:
        """Identify files that need new embeddings."""
        new_files = []
        
        for filename, content, content_hash in html_files:
            # Check if file is new or content has changed
            if (filename not in cache["content_hashes"] or 
                cache["content_hashes"][filename] != content_hash):
                new_files.append((filename, content))
                logger.info(f"🆕 New or changed file: {filename}")
            else:
                logger.info(f"✅ Using cached embedding: {filename}")
        
        return new_files
    
    def create_embeddings_for_new_files(self, new_files: List[Tuple[str, str]]) -> Dict[str, List[float]]:
        """Create embeddings only for new files."""
        new_embeddings = {}
        
        if not new_files:
            logger.info("No new files to embed!")
            return new_embeddings
        
        logger.info(f"Creating embeddings for {len(new_files)} new files...")
        
        for i, (filename, content) in enumerate(new_files):
            logger.info(f"Processing new file {i+1}/{len(new_files)}: {filename}")
            
            try:
                result = self.voyage_client.embed(
                    [content],
                    model=self.model,
                    input_type="document"
                )
                
                new_embeddings[filename] = result.embeddings[0]
                logger.info(f"✓ Successfully embedded: {filename}")
                
                # Add delay between requests to respect rate limits
                if i < len(new_files) - 1:
                    logger.info("Waiting 25 seconds for rate limit...")
                    time.sleep(25)
                    
            except Exception as e:
                logger.error(f"Error embedding {filename}: {e}")
                # Add a longer delay if we hit rate limits
                if "rate limit" in str(e).lower():
                    logger.info("Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                else:
                    raise e
        
        return new_embeddings
    
    def combine_embeddings(self, cache: Dict, new_embeddings: Dict[str, List[float]], 
                          html_files: List[Tuple[str, str, str]]) -> Tuple[np.ndarray, List[str]]:
        """Combine cached and new embeddings."""
        all_embeddings = []
        all_filenames = []
        
        # Process all files in order they were read
        for filename, content, content_hash in html_files:
            if filename in new_embeddings:
                # Use new embedding
                all_embeddings.append(new_embeddings[filename])
                all_filenames.append(filename)
                logger.info(f"Using new embedding for: {filename}")
            elif filename in cache["embeddings"]:
                # Use cached embedding
                all_embeddings.append(cache["embeddings"][filename])
                all_filenames.append(filename)
                logger.info(f"Using cached embedding for: {filename}")
            else:
                logger.warning(f"No embedding found for: {filename}")
        
        return np.array(all_embeddings), all_filenames
    
    def update_cache(self, cache: Dict, new_embeddings: Dict[str, List[float]], 
                    html_files: List[Tuple[str, str, str]]):
        """Update cache with new embeddings and hashes."""
        # Update embeddings
        cache["embeddings"].update(new_embeddings)
        
        # Update content hashes
        for filename, content, content_hash in html_files:
            cache["content_hashes"][filename] = content_hash
        
        # Update filenames list
        cache["filenames"] = [filename for filename, _, _ in html_files]
        
        logger.info(f"Updated cache with {len(new_embeddings)} new embeddings")
    
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
        """Run the incremental embedding pipeline."""
        logger.info("Starting incremental HTML embedding and visualization pipeline...")
        
        # Step 1: Load existing cache
        cache = self.load_cache()
        
        # Step 2: Read all HTML files
        html_files = self.read_html_files(directory)
        if not html_files:
            logger.error("No HTML files found to process!")
            return
        
        # Step 3: Identify new files that need embedding
        new_files = self.identify_new_files(html_files, cache)
        
        # Step 4: Create embeddings for new files only
        if new_files:
            estimated_minutes = len(new_files) * 25 / 60
            logger.info(f"⏱️  Estimated processing time: {estimated_minutes:.1f} minutes for {len(new_files)} new files")
            
            new_embeddings = self.create_embeddings_for_new_files(new_files)
            
            # Step 5: Update cache
            self.update_cache(cache, new_embeddings, html_files)
            self.save_cache(cache)
        else:
            logger.info("No new files to process, using all cached embeddings!")
        
        # Step 6: Combine all embeddings
        embeddings, filenames = self.combine_embeddings(cache, {}, html_files)
        
        # Step 7: Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(embeddings)
        
        # Step 8: Create visualization
        self.create_visualization(reduced_embeddings, filenames)
        
        logger.info("Incremental pipeline completed successfully!")


def main():
    """Main function to run the incremental embedding pipeline."""
    
    # Initialize and run the visualizer
    try:
        visualizer = IncrementalHTMLEmbeddingVisualizer(
            api_key=None,  # Will use VOYAGE_API_KEY environment variable
            model="voyage-code-2"
        )
        
        visualizer.run()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 