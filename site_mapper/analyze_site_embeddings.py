#!/usr/bin/env python3
"""
analyze_site_embeddings.py
--------------------------
Analyzes structural and visual embeddings from crawled website data.

Takes results from website_mapper.py and generates:
1. Structural similarity analysis (DOM, CSS, forms, links)
2. Visual similarity analysis (CLIP embeddings of screenshots)
3. Combined visualizations and clustering

Usage:
-----
python analyze_site_embeddings.py results/docs_stripe_com/
python analyze_site_embeddings.py results/nike/ --output-dir analysis_output/
"""

# Set numba/OpenMP environment before any imports
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # fallback if needed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# For visual embeddings
try:
    import torch
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install torch torchvision clip-by-openai")

# For HTML parsing
from bs4 import BeautifulSoup

class SiteEmbeddingAnalyzer:
    def __init__(self, site_results_path: Path, output_dir: Path = None):
        self.site_results_path = Path(site_results_path)
        self.output_dir = output_dir or (self.site_results_path / "embeddings_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for embeddings and metadata
        self.pages = {}
        self.structural_embeddings = {}
        self.visual_embeddings = {}
        self.combined_embeddings = {}
        self.page_metadata = {}
        
        # CLIP model for visual embeddings
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
            self.clip_model.eval()
    
    def discover_pages(self) -> List[Dict]:
        """Discover all page directories in the site results"""
        pages = []
        
        for page_dir in self.site_results_path.iterdir():
            if page_dir.is_dir() and page_dir.name != "embeddings_analysis":
                html_files = list(page_dir.glob("*.html"))
                screenshot_files = list(page_dir.glob("*.png"))
                
                if html_files or screenshot_files:
                    page_info = {
                        "page_id": page_dir.name,
                        "page_dir": page_dir,
                        "html_files": html_files,
                        "screenshot_files": screenshot_files,
                        "url": self._extract_url_from_page_id(page_dir.name)
                    }
                    pages.append(page_info)
                    self.pages[page_dir.name] = page_info
        
        print(f"Discovered {len(pages)} pages for analysis")
        return pages
    
    def _extract_url_from_page_id(self, page_id: str) -> str:
        """Extract original URL from page ID (reverse of URL-to-filename conversion)"""
        # Convert back from filename format to URL
        url = page_id.replace("___", "://").replace("__", "/")
        return url
    
    def extract_structural_features(self, html_content: str) -> Dict[str, Any]:
        """Extract structural features from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Basic DOM structure
        all_tags = [tag.name for tag in soup.find_all()]
        tag_counts = Counter(all_tags)
        
        # CSS classes and IDs
        classes = []
        ids = []
        for tag in soup.find_all():
            if tag.get('class'):
                classes.extend(tag['class'])
            if tag.get('id'):
                ids.append(tag['id'])
        
        class_counts = Counter(classes)
        id_counts = Counter(ids)
        
        # Forms and inputs
        forms = soup.find_all('form')
        inputs = soup.find_all('input')
        input_types = [inp.get('type', 'text') for inp in inputs]
        
        # Links and navigation
        links = soup.find_all('a', href=True)
        internal_links = []
        external_links = []
        
        for link in links:
            href = link['href']
            if href.startswith('http'):
                external_links.append(href)
            else:
                internal_links.append(href)
        
        # Text content features
        text = soup.get_text()
        text_length = len(text)
        word_count = len(text.split())
        
        # Semantic HTML5 elements
        semantic_tags = ['header', 'nav', 'main', 'article', 'section', 'aside', 'footer']
        semantic_counts = {tag: len(soup.find_all(tag)) for tag in semantic_tags}
        
        # Meta information
        meta_tags = soup.find_all('meta')
        meta_info = {}
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                meta_info[name] = content
        
        return {
            # Basic structure
            'total_tags': len(all_tags),
            'unique_tags': len(set(all_tags)),
            'tag_diversity': len(set(all_tags)) / len(all_tags) if all_tags else 0,
            
            # Most common tags (normalized)
            'div_ratio': tag_counts.get('div', 0) / len(all_tags) if all_tags else 0,
            'span_ratio': tag_counts.get('span', 0) / len(all_tags) if all_tags else 0,
            'p_ratio': tag_counts.get('p', 0) / len(all_tags) if all_tags else 0,
            'img_ratio': tag_counts.get('img', 0) / len(all_tags) if all_tags else 0,
            
            # CSS and styling
            'total_classes': len(classes),
            'unique_classes': len(set(classes)),
            'total_ids': len(ids),
            'unique_ids': len(set(ids)),
            
            # Interactive elements
            'form_count': len(forms),
            'input_count': len(inputs),
            'button_count': tag_counts.get('button', 0),
            'select_count': tag_counts.get('select', 0),
            
            # Navigation and links
            'internal_links': len(internal_links),
            'external_links': len(external_links),
            'nav_elements': tag_counts.get('nav', 0),
            
            # Content
            'text_length': text_length,
            'word_count': word_count,
            'heading_count': sum(tag_counts.get(f'h{i}', 0) for i in range(1, 7)),
            
            # Semantic structure
            **{f'semantic_{tag}': count for tag, count in semantic_counts.items()},
            
            # Page type indicators
            'has_table': tag_counts.get('table', 0) > 0,
            'has_video': tag_counts.get('video', 0) > 0,
            'has_audio': tag_counts.get('audio', 0) > 0,
            'has_canvas': tag_counts.get('canvas', 0) > 0,
            'has_svg': tag_counts.get('svg', 0) > 0,
            
            # Raw counts for top elements (for detailed analysis)
            'raw_tag_counts': dict(tag_counts.most_common(20)),
            'raw_class_counts': dict(class_counts.most_common(10)),
            'meta_info': meta_info
        }
    
    def generate_structural_embeddings(self):
        """Generate structural embeddings for all pages"""
        print("Generating structural embeddings...")
        
        feature_data = []
        page_ids = []
        
        for page_id, page_info in self.pages.items():
            # Use the main HTML file (prefer page.html, fallback to crawled.html)
            html_file = None
            for html_path in page_info['html_files']:
                if html_path.name == 'page.html':
                    html_file = html_path
                    break
            
            if not html_file and page_info['html_files']:
                html_file = page_info['html_files'][0]
            
            if html_file and html_file.exists():
                try:
                    html_content = html_file.read_text(encoding='utf-8')
                    features = self.extract_structural_features(html_content)
                    
                    # Store metadata
                    self.page_metadata[page_id] = {
                        'url': page_info['url'],
                        'html_file': str(html_file),
                        'features': features
                    }
                    
                    # Extract numeric features for embedding
                    numeric_features = {k: v for k, v in features.items() 
                                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    
                    feature_data.append(numeric_features)
                    page_ids.append(page_id)
                    
                except Exception as e:
                    print(f"Error processing {page_id}: {e}")
        
        if not feature_data:
            print("No structural features extracted")
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(feature_data, index=page_ids)
        df = df.fillna(0)  # Fill NaN values with 0
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(df)
        
        # Store embeddings
        for i, page_id in enumerate(page_ids):
            self.structural_embeddings[page_id] = normalized_features[i]
        
        print(f"Generated structural embeddings for {len(page_ids)} pages")
        
        # Save feature analysis
        feature_analysis = {
            'feature_names': list(df.columns),
            'feature_stats': df.describe().to_dict(),
            'page_features': df.to_dict('index')
        }
        
        with open(self.output_dir / 'structural_features.json', 'w') as f:
            json.dump(feature_analysis, f, indent=2, default=str)
    
    def generate_visual_embeddings(self):
        """Generate visual embeddings using CLIP"""
        if not CLIP_AVAILABLE:
            print("CLIP not available, skipping visual embeddings")
            return
        
        print("Generating visual embeddings...")
        
        for page_id, page_info in self.pages.items():
            screenshot_files = page_info['screenshot_files']
            
            if not screenshot_files:
                continue
            
            # Use the first screenshot file
            screenshot_path = screenshot_files[0]
            
            try:
                # Load and preprocess image
                image = Image.open(screenshot_path).convert('RGB')
                image_input = self.clip_preprocess(image).unsqueeze(0)
                
                # Generate embedding
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Store embedding
                self.visual_embeddings[page_id] = image_features.cpu().numpy().flatten()
                
            except Exception as e:
                print(f"Error generating visual embedding for {page_id}: {e}")
        
        print(f"Generated visual embeddings for {len(self.visual_embeddings)} pages")
    
    def generate_combined_embeddings(self):
        """Generate combined embeddings from structural and visual data"""
        if not self.structural_embeddings or not self.visual_embeddings:
            print("Cannot generate combined embeddings: missing structural or visual data")
            return
        
        print("Generating combined embeddings...")
        
        # Find pages that have both structural and visual embeddings
        common_pages = set(self.structural_embeddings.keys()) & set(self.visual_embeddings.keys())
        
        if not common_pages:
            print("No pages have both structural and visual embeddings")
            return
        
        self.combined_embeddings = {}
        
        # Normalize embeddings to same scale before combining
        structural_data = np.array([self.structural_embeddings[page] for page in common_pages])
        visual_data = np.array([self.visual_embeddings[page] for page in common_pages])
        
        # Standardize both embedding types
        struct_scaler = StandardScaler()
        visual_scaler = StandardScaler()
        
        structural_normalized = struct_scaler.fit_transform(structural_data)
        visual_normalized = visual_scaler.fit_transform(visual_data)
        
        # Combine embeddings - you can experiment with different combination strategies
        for i, page in enumerate(common_pages):
            # Strategy 1: Concatenation (gives equal weight to both)
            combined = np.concatenate([structural_normalized[i], visual_normalized[i]])
            
            # Strategy 2: Weighted average (uncomment to use instead)
            # structural_weight = 0.6  # Adjust weights as needed
            # visual_weight = 0.4
            # # Need to make dimensions compatible for weighted average
            # min_dims = min(len(structural_normalized[i]), len(visual_normalized[i]))
            # struct_truncated = structural_normalized[i][:min_dims]
            # visual_truncated = visual_normalized[i][:min_dims]
            # combined = structural_weight * struct_truncated + visual_weight * visual_truncated
            
            self.combined_embeddings[page] = combined
        
        print(f"Generated combined embeddings for {len(self.combined_embeddings)} pages")
    
    def compute_similarity_matrices(self):
        """Compute similarity matrices for structural and visual embeddings"""
        results = {}
        
        # Structural similarity
        if self.structural_embeddings:
            page_ids = list(self.structural_embeddings.keys())
            embeddings_matrix = np.array([self.structural_embeddings[pid] for pid in page_ids])
            structural_sim = cosine_similarity(embeddings_matrix)
            
            results['structural'] = {
                'page_ids': page_ids,
                'similarity_matrix': structural_sim,
                'embeddings': embeddings_matrix
            }
        
        # Visual similarity
        if self.visual_embeddings:
            page_ids = list(self.visual_embeddings.keys())
            embeddings_matrix = np.array([self.visual_embeddings[pid] for pid in page_ids])
            visual_sim = cosine_similarity(embeddings_matrix)
            
            results['visual'] = {
                'page_ids': page_ids,
                'similarity_matrix': visual_sim,
                'embeddings': embeddings_matrix
            }
        
        # Combined similarity
        if self.combined_embeddings:
            page_ids = list(self.combined_embeddings.keys())
            embeddings_matrix = np.array([self.combined_embeddings[pid] for pid in page_ids])
            combined_sim = cosine_similarity(embeddings_matrix)
            
            results['combined'] = {
                'page_ids': page_ids,
                'similarity_matrix': combined_sim,
                'embeddings': embeddings_matrix
            }
        
        return results
    
    def create_visualizations(self, similarity_results):
        """Create similarity visualizations"""
        # Determine grid size based on available data
        has_structural = 'structural' in similarity_results
        has_visual = 'visual' in similarity_results
        has_combined = 'combined' in similarity_results
        
        if has_combined:
            fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
        fig.suptitle(f'Site Analysis: {self.site_results_path.name}', fontsize=16)
        
        # Structural similarity heatmap
        if has_structural:
            structural_data = similarity_results['structural']
            ax = axes[0, 0]
            
            sns.heatmap(
                structural_data['similarity_matrix'],
                xticklabels=[pid[:20] + '...' if len(pid) > 20 else pid for pid in structural_data['page_ids']],
                yticklabels=[pid[:20] + '...' if len(pid) > 20 else pid for pid in structural_data['page_ids']],
                annot=True,
                fmt='.2f',
                cmap='viridis',
                ax=ax
            )
            ax.set_title('Structural Similarity')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
        
        # Visual similarity heatmap
        if has_visual:
            visual_data = similarity_results['visual']
            ax = axes[0, 1]
            
            sns.heatmap(
                visual_data['similarity_matrix'],
                xticklabels=[pid[:20] + '...' if len(pid) > 20 else pid for pid in visual_data['page_ids']],
                yticklabels=[pid[:20] + '...' if len(pid) > 20 else pid for pid in visual_data['page_ids']],
                annot=True,
                fmt='.2f',
                cmap='plasma',
                ax=ax
            )
            ax.set_title('Visual Similarity')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
        
        # Combined similarity heatmap
        if has_combined:
            combined_data = similarity_results['combined']
            ax = axes[0, 2]
            
            sns.heatmap(
                combined_data['similarity_matrix'],
                xticklabels=[pid[:20] + '...' if len(pid) > 20 else pid for pid in combined_data['page_ids']],
                yticklabels=[pid[:20] + '...' if len(pid) > 20 else pid for pid in combined_data['page_ids']],
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                ax=ax
            )
            ax.set_title('Combined Similarity (Structural + Visual)')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
        
        # PCA embeddings for structural data
        if has_structural:
            structural_data = similarity_results['structural']
            ax = axes[1, 0]
            
            if len(structural_data['embeddings']) > 2:
                reducer = (umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=max(2, min(10, structural_data['embeddings'].shape[0] - 1)),
                    n_epochs=200,
                    metric='cosine')
                    if UMAP_AVAILABLE else PCA(n_components=2))
                embedding_2d = reducer.fit_transform(structural_data['embeddings'])
                
                scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=range(len(structural_data['page_ids'])), 
                                   cmap='tab10', s=100, alpha=0.7)
                
                # Add labels
                for i, page_id in enumerate(structural_data['page_ids']):
                    label = page_id[:15] + '...' if len(page_id) > 15 else page_id
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                ax.set_title('Structural Embeddings (PCA)')
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                if UMAP_AVAILABLE:
                    ax.set_title('Structural Embeddings (UMAP)')
                    ax.set_xlabel('UMAP 1')
                    ax.set_ylabel('UMAP 2')
        
        # PCA embeddings for visual data
        if has_visual:
            visual_data = similarity_results['visual']
            ax = axes[1, 1]
            
            if len(visual_data['embeddings']) > 2:
                reducer = (umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=max(2, min(10, visual_data['embeddings'].shape[0] - 1)),
                    n_epochs=200,
                    metric='cosine')
                    if UMAP_AVAILABLE else PCA(n_components=2))
                embedding_2d = reducer.fit_transform(visual_data['embeddings'])
                
                scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=range(len(visual_data['page_ids'])), 
                                   cmap='tab10', s=100, alpha=0.7)
                
                # Add labels
                for i, page_id in enumerate(visual_data['page_ids']):
                    label = page_id[:15] + '...' if len(page_id) > 15 else page_id
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                ax.set_title('Visual Embeddings (PCA)')
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                if UMAP_AVAILABLE:
                    ax.set_title('Visual Embeddings (UMAP)')
                    ax.set_xlabel('UMAP 1')
                    ax.set_ylabel('UMAP 2')
        
        # PCA embeddings for combined data
        if has_combined:
            combined_data = similarity_results['combined']
            ax = axes[1, 2]
            
            if len(combined_data['embeddings']) > 2:
                reducer = (umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=max(2, min(10, combined_data['embeddings'].shape[0] - 1)),
                    n_epochs=200,
                    metric='cosine')
                    if UMAP_AVAILABLE else PCA(n_components=2))
                embedding_2d = reducer.fit_transform(combined_data['embeddings'])
                
                scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=range(len(combined_data['page_ids'])), 
                                   cmap='tab10', s=100, alpha=0.7)
                
                # Add labels
                for i, page_id in enumerate(combined_data['page_ids']):
                    label = page_id[:15] + '...' if len(page_id) > 15 else page_id
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                ax.set_title('Combined Embeddings (PCA)')
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                if UMAP_AVAILABLE:
                    ax.set_title('Combined Embeddings (UMAP)')
                    ax.set_xlabel('UMAP 1')
                    ax.set_ylabel('UMAP 2')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'site_similarity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {self.output_dir / 'site_similarity_analysis.png'}")
    
    def generate_similarity_report(self, similarity_results):
        """Generate a detailed similarity report"""
        report = {
            'site_name': self.site_results_path.name,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_pages': len(self.pages),
            'pages_with_structural_data': len(self.structural_embeddings),
            'pages_with_visual_data': len(self.visual_embeddings),
            'pages_with_combined_data': len(self.combined_embeddings),
        }
        
        # Top similar page pairs
        for analysis_type in ['structural', 'visual', 'combined']:
            if analysis_type in similarity_results:
                data = similarity_results[analysis_type]
                sim_matrix = data['similarity_matrix']
                page_ids = data['page_ids']
                
                # Find most similar pairs (excluding self-similarity)
                similar_pairs = []
                for i in range(len(page_ids)):
                    for j in range(i + 1, len(page_ids)):
                        similarity = sim_matrix[i, j]
                        similar_pairs.append({
                            'page1': page_ids[i],
                            'page2': page_ids[j],
                            'similarity': float(similarity),
                            'url1': self.pages[page_ids[i]]['url'],
                            'url2': self.pages[page_ids[j]]['url']
                        })
                
                # Sort by similarity
                similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
                
                report[f'{analysis_type}_top_similar'] = similar_pairs[:10]
                report[f'{analysis_type}_avg_similarity'] = float(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]))
        
        # Save report
        with open(self.output_dir / 'similarity_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Similarity report saved to {self.output_dir / 'similarity_report.json'}")
        return report
    
    def run_full_analysis(self):
        """Run complete embedding analysis pipeline"""
        print(f"Starting analysis of {self.site_results_path}")
        
        # Discover pages
        pages = self.discover_pages()
        if not pages:
            print("No pages found for analysis")
            return
        
        # Generate embeddings
        self.generate_structural_embeddings()
        self.generate_visual_embeddings()
        self.generate_combined_embeddings()
        
        # Compute similarities
        similarity_results = self.compute_similarity_matrices()
        
        # Create visualizations
        self.create_visualizations(similarity_results)
        
        # Generate report
        report = self.generate_similarity_report(similarity_results)
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}")
        print(f"- Structural embeddings: {len(self.structural_embeddings)} pages")
        print(f"- Visual embeddings: {len(self.visual_embeddings)} pages")
        print(f"- Combined embeddings: {len(self.combined_embeddings)} pages")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Analyze structural and visual embeddings from crawled website data')
    parser.add_argument('site_results_path', help='Path to site results directory (e.g., results/docs_stripe_com/)')
    parser.add_argument('--output-dir', help='Output directory for analysis results')
    args = parser.parse_args()
    
    site_results_path = Path(args.site_results_path)
    if not site_results_path.exists():
        print(f"Error: Site results path {site_results_path} does not exist")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    analyzer = SiteEmbeddingAnalyzer(site_results_path, output_dir)
    analyzer.run_full_analysis()

if __name__ == '__main__':
    main() 