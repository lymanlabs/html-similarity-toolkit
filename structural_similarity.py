#!/usr/bin/env python3
"""
HTML Structural Similarity Analysis

This script analyzes HTML files based on their structural properties rather than
semantic content. It uses a hybrid approach:
1. Rule-based page type classification
2. Structural similarity metrics within page types
3. Visualization of structural clusters
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import difflib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
import networkx as nx


class StructuralAnalyzer:
    """Analyzes HTML files for structural similarity rather than semantic content."""
    
    def __init__(self, html_dir: str = "sample_html"):
        self.html_dir = Path(html_dir)
        self.html_files = {}
        self.page_types = {}
        self.structural_features = {}
        self.similarity_matrix = None
        
    def load_html_files(self):
        """Load all HTML files from the directory."""
        print(f"\n📁 Loading HTML files from {self.html_dir}...")
        
        for file_path in self.html_dir.glob("*.html"):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.html_files[file_path.name] = f.read()
                
        print(f"✓ Loaded {len(self.html_files)} HTML files")
        
    def classify_page_types(self):
        """Classify each HTML file into a page type based on structural patterns."""
        print("\n🏷️  Classifying page types...")
        
        for filename, html in self.html_files.items():
            soup = BeautifulSoup(html, 'html.parser')
            page_type = self._detect_page_type(soup)
            self.page_types[filename] = page_type
            print(f"  {filename}: {page_type}")
            
        # Summary
        type_counts = Counter(self.page_types.values())
        print("\n📊 Page type distribution:")
        for ptype, count in type_counts.items():
            print(f"  {ptype}: {count} pages")
            
    def _detect_page_type(self, soup: BeautifulSoup) -> str:
        """
        Detect page type using rule-based heuristics.
        Optimized for Adidas website structure.
        """
        # Check for product page indicators (PDPs)
        # Adidas uses data-auto-id="add-to-bag" on product pages
        if (soup.find(attrs={'data-auto-id': 'add-to-bag'}) or
            soup.find('button', attrs={'title': re.compile('add to bag', re.I)})):
            return 'product'
            
        # Check for cart page indicators
        # Look for cart-specific headers or sections
        cart_headers = ['your bag', 'shopping bag', 'cart summary', 'bag summary']
        if (soup.find(['h1', 'h2'], string=re.compile('|'.join(cart_headers), re.I)) or
            soup.find(attrs={'class': re.compile('cart-content|bag-item|cart-item', re.I)})):
            return 'cart'
            
        # Check for checkout indicators
        checkout_patterns = ['checkout', 'payment', 'billing', 'shipping', 'delivery-address']
        if (any(soup.find(attrs={'class': re.compile(pattern, re.I)}) for pattern in checkout_patterns) or
            soup.find(['h1', 'h2'], string=re.compile('checkout|payment|shipping', re.I))):
            return 'checkout'
            
        # Check for homepage indicators
        # Typically has many navigation links and promotional sections
        nav_count = len(soup.find_all('nav'))
        link_count = len(soup.find_all('a'))
        section_count = len(soup.find_all(['section', 'article']))
        
        if nav_count >= 1 and link_count > 50 and section_count > 3:
            return 'homepage'
            
        # Check for category/listing pages
        # These typically have product grids or lists
        if (soup.find(attrs={'class': re.compile('product-grid|product-list|plp-grid', re.I)}) or
            len(soup.find_all(attrs={'class': re.compile('product-card|product-tile', re.I)})) > 3):
            return 'category'
            
        # Check for contact page indicators
        contact_patterns = ['contact', 'get-in-touch', 'customer-service']
        if (any(soup.find(attrs={'class': re.compile(pattern, re.I)}) for pattern in contact_patterns) or
            soup.find(['h1', 'h2'], string=re.compile('contact|help|customer service', re.I))):
            return 'contact'
            
        # Check for account/profile pages
        if (soup.find(attrs={'class': re.compile('account|profile|my-account', re.I)}) or
            soup.find(['h1', 'h2'], string=re.compile('my account|profile|account', re.I))):
            return 'account'
            
        return 'other'
        
    def extract_structural_features(self):
        """Extract various structural features from each HTML file."""
        print("\n🔍 Extracting structural features...")
        
        for filename, html in self.html_files.items():
            soup = BeautifulSoup(html, 'html.parser')
            
            features = {
                'dom_structure': self._get_dom_structure(soup),
                'tag_counts': self._get_tag_counts(soup),
                'css_fingerprint': self._get_css_fingerprint(soup),
                'form_structure': self._get_form_structure(soup),
                'link_structure': self._get_link_structure(soup),
                'depth_stats': self._get_depth_statistics(soup),
                'id_classes': self._get_id_class_patterns(soup)
            }
            
            self.structural_features[filename] = features
            
    def _get_dom_structure(self, soup: BeautifulSoup) -> List[str]:
        """Get a simplified representation of DOM structure."""
        structure = []
        for tag in soup.find_all()[:200]:  # Limit to first 200 tags for efficiency
            structure.append(tag.name)
        return structure
        
    def _get_tag_counts(self, soup: BeautifulSoup) -> Dict[str, int]:
        """Count occurrences of each tag type."""
        return Counter(tag.name for tag in soup.find_all())
        
    def _get_css_fingerprint(self, soup: BeautifulSoup) -> Dict[str, any]:
        """Extract CSS class and ID patterns."""
        classes = []
        ids = []
        
        for tag in soup.find_all():
            if tag.get('class'):
                classes.extend(tag['class'])
            if tag.get('id'):
                ids.append(tag['id'])
                
        return {
            'unique_classes': len(set(classes)),
            'total_classes': len(classes),
            'unique_ids': len(set(ids)),
            'common_classes': Counter(classes).most_common(10)
        }
        
    def _get_form_structure(self, soup: BeautifulSoup) -> Dict[str, any]:
        """Analyze form elements."""
        forms = soup.find_all('form')
        return {
            'form_count': len(forms),
            'input_count': len(soup.find_all('input')),
            'button_count': len(soup.find_all('button')),
            'select_count': len(soup.find_all('select'))
        }
        
    def _get_link_structure(self, soup: BeautifulSoup) -> Dict[str, any]:
        """Analyze link patterns."""
        links = soup.find_all('a', href=True)
        internal_links = [l for l in links if not l['href'].startswith(('http://', 'https://'))]
        external_links = [l for l in links if l['href'].startswith(('http://', 'https://'))]
        
        return {
            'total_links': len(links),
            'internal_links': len(internal_links),
            'external_links': len(external_links),
            'nav_links': len(soup.find_all('nav'))
        }
        
    def _get_depth_statistics(self, soup: BeautifulSoup) -> Dict[str, any]:
        """Calculate DOM tree depth statistics."""
        def get_depth(tag):
            depth = 0
            while tag.parent:
                depth += 1
                tag = tag.parent
            return depth
            
        depths = [get_depth(tag) for tag in soup.find_all()[:100]]
        
        return {
            'max_depth': max(depths) if depths else 0,
            'avg_depth': np.mean(depths) if depths else 0,
            'depth_variance': np.var(depths) if depths else 0
        }
        
    def _get_id_class_patterns(self, soup: BeautifulSoup) -> List[str]:
        """Extract patterns from IDs and classes for structural comparison."""
        patterns = []
        
        # Look for common structural patterns
        for tag in soup.find_all()[:100]:
            if tag.get('id'):
                # Add ID patterns (removing specific identifiers)
                pattern = re.sub(r'\d+', 'N', tag['id'])
                patterns.append(f"id:{pattern}")
                
            if tag.get('class'):
                # Add class combinations
                class_combo = '.'.join(sorted(tag['class']))
                patterns.append(f"class:{class_combo}")
                
        return patterns[:50]  # Limit to top patterns
        
    def calculate_structural_similarity(self):
        """Calculate pairwise structural similarity between documents."""
        print("\n📐 Calculating structural similarity...")
        
        filenames = list(self.html_files.keys())
        n = len(filenames)
        self.similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                sim = self._calculate_pairwise_similarity(
                    self.structural_features[filenames[i]],
                    self.structural_features[filenames[j]]
                )
                self.similarity_matrix[i, j] = sim
                
    def _calculate_pairwise_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two sets of structural features."""
        similarities = []
        
        # DOM structure similarity
        dom_sim = difflib.SequenceMatcher(
            None, 
            features1['dom_structure'][:100], 
            features2['dom_structure'][:100]
        ).ratio()
        similarities.append(dom_sim * 2)  # Weight DOM structure heavily
        
        # Tag count similarity
        tags1 = features1['tag_counts']
        tags2 = features2['tag_counts']
        all_tags = set(tags1.keys()) | set(tags2.keys())
        
        if all_tags:
            tag_vectors = []
            for tags in [tags1, tags2]:
                vector = [tags.get(tag, 0) for tag in sorted(all_tags)]
                tag_vectors.append(vector)
            
            tag_sim = cosine_similarity([tag_vectors[0]], [tag_vectors[1]])[0, 0]
            similarities.append(tag_sim)
        
        # CSS fingerprint similarity
        css1 = features1['css_fingerprint']
        css2 = features2['css_fingerprint']
        
        css_metrics = [
            abs(css1['unique_classes'] - css2['unique_classes']) / max(css1['unique_classes'], css2['unique_classes'], 1),
            abs(css1['unique_ids'] - css2['unique_ids']) / max(css1['unique_ids'], css2['unique_ids'], 1)
        ]
        css_sim = 1 - np.mean(css_metrics)
        similarities.append(css_sim)
        
        # Form structure similarity
        form1 = features1['form_structure']
        form2 = features2['form_structure']
        
        form_diffs = [
            abs(form1[k] - form2[k]) / max(form1[k], form2[k], 1)
            for k in form1.keys()
        ]
        form_sim = 1 - np.mean(form_diffs)
        similarities.append(form_sim)
        
        # ID/Class pattern similarity
        patterns1 = set(features1['id_classes'])
        patterns2 = set(features2['id_classes'])
        
        if patterns1 or patterns2:
            pattern_sim = len(patterns1 & patterns2) / len(patterns1 | patterns2)
            similarities.append(pattern_sim * 1.5)  # Weight patterns
        
        return np.mean(similarities)
        
    def visualize_structural_clusters(self):
        """Create visualizations of structural similarity."""
        print("\n📊 Creating visualizations...")
        
        # Create a figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        filenames = list(self.html_files.keys())
        
        # 1. Heatmap of similarity matrix
        sns.heatmap(
            self.similarity_matrix,
            xticklabels=filenames,
            yticklabels=filenames,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            ax=ax1
        )
        ax1.set_title('Structural Similarity Matrix', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # 2. MDS visualization colored by page type
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        distance_matrix = 1 - self.similarity_matrix
        coords = mds.fit_transform(distance_matrix)
        
        # Color by page type
        page_type_colors = {
            'product': '#FF6B6B',
            'cart': '#4ECDC4',
            'checkout': '#45B7D1',
            'homepage': '#96CEB4',
            'contact': '#DDA0DD',
            'form-page': '#FFD93D',
            'other': '#95A5A6'
        }
        
        colors = [page_type_colors.get(self.page_types[f], '#95A5A6') for f in filenames]
        
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
            
        # Add legend for page types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=10, label=ptype)
            for ptype, color in page_type_colors.items()
            if ptype in self.page_types.values()
        ]
        ax2.legend(handles=legend_elements, loc='best')
        ax2.set_title('Structural Similarity (MDS) - Colored by Page Type', fontsize=14)
        ax2.set_xlabel('MDS Dimension 1')
        ax2.set_ylabel('MDS Dimension 2')
        
        # 3. Page type distribution
        type_counts = Counter(self.page_types.values())
        ax3.bar(type_counts.keys(), type_counts.values(), 
                color=[page_type_colors.get(t, '#95A5A6') for t in type_counts.keys()])
        ax3.set_title('Page Type Distribution', fontsize=14)
        ax3.set_xlabel('Page Type')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Within-type similarity analysis
        within_type_similarities = {}
        
        for page_type in set(self.page_types.values()):
            pages_of_type = [f for f, t in self.page_types.items() if t == page_type]
            
            if len(pages_of_type) > 1:
                indices = [filenames.index(p) for p in pages_of_type]
                sims = []
                
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        sims.append(self.similarity_matrix[indices[i], indices[j]])
                        
                within_type_similarities[page_type] = sims
        
        # Box plot of within-type similarities
        if within_type_similarities:
            data_for_plot = []
            labels_for_plot = []
            
            for ptype, sims in within_type_similarities.items():
                data_for_plot.extend(sims)
                labels_for_plot.extend([ptype] * len(sims))
            
            import pandas as pd
            df = pd.DataFrame({'Page Type': labels_for_plot, 'Similarity': data_for_plot})
            
            # Sort by median similarity
            sorted_types = df.groupby('Page Type')['Similarity'].median().sort_values(ascending=False).index
            
            sns.boxplot(
                data=df,
                x='Page Type',
                y='Similarity',
                order=sorted_types,
                palette=[page_type_colors.get(t, '#95A5A6') for t in sorted_types],
                ax=ax4
            )
            ax4.set_title('Within-Type Structural Similarity', fontsize=14)
            ax4.set_ylabel('Structural Similarity Score')
            ax4.tick_params(axis='x', rotation=45)
            ax4.set_ylim(0, 1)
            
        plt.tight_layout()
        plt.savefig('structural_similarity_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to structural_similarity_analysis.png")
        
        # Print specific insights about PDPs
        self._print_pdp_analysis(filenames)
        
    def _print_pdp_analysis(self, filenames):
        """Print specific analysis about product pages."""
        print("\n🔍 Product Page Analysis:")
        
        product_pages = [f for f, t in self.page_types.items() if t == 'product']
        
        if len(product_pages) >= 2:
            print(f"\nFound {len(product_pages)} product pages: {product_pages}")
            
            # Check similarity between product pages
            indices = [filenames.index(p) for p in product_pages]
            
            print("\nPairwise structural similarity between product pages:")
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    sim = self.similarity_matrix[indices[i], indices[j]]
                    print(f"  {product_pages[i]} <-> {product_pages[j]}: {sim:.3f}")
                    
            # Compare with other page types
            if len(product_pages) >= 2 and 'cart' in self.page_types.values():
                cart_pages = [f for f, t in self.page_types.items() if t == 'cart']
                if cart_pages:
                    cart_idx = filenames.index(cart_pages[0])
                    prod_idx = indices[0]
                    cross_sim = self.similarity_matrix[prod_idx, cart_idx]
                    print(f"\n  {product_pages[0]} <-> {cart_pages[0]}: {cross_sim:.3f}")
                    print("  (Product vs Cart - expect lower similarity)")
        
    def save_results(self):
        """Save analysis results to JSON."""
        results = {
            'page_types': self.page_types,
            'summary': {
                'total_files': len(self.html_files),
                'page_type_distribution': dict(Counter(self.page_types.values())),
                'average_similarity': float(np.mean(self.similarity_matrix)),
                'structural_features_sample': {
                    filename: {
                        'tag_count': len(features['tag_counts']),
                        'unique_classes': features['css_fingerprint']['unique_classes'],
                        'form_count': features['form_structure']['form_count'],
                        'max_depth': features['depth_stats']['max_depth']
                    }
                    for filename, features in list(self.structural_features.items())[:3]
                }
            }
        }
        
        with open('structural_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print("\n✓ Saved analysis results to structural_analysis_results.json")
        
    def run_analysis(self):
        """Run the complete structural analysis pipeline."""
        self.load_html_files()
        self.classify_page_types()
        self.extract_structural_features()
        self.calculate_structural_similarity()
        self.visualize_structural_clusters()
        self.save_results()


def main():
    """Main execution function."""
    print("=" * 60)
    print("HTML STRUCTURAL SIMILARITY ANALYSIS")
    print("=" * 60)
    
    analyzer = StructuralAnalyzer()
    analyzer.run_analysis()
    
    print("\n✅ Analysis complete!")
    print("\nKey insights:")
    print("- Page types are classified based on structural patterns")
    print("- Similarity is calculated using DOM structure, tags, CSS, and forms")
    print("- Visualizations show how pages cluster by structure, not content")
    print("- Product pages should show high mutual similarity if they share templates")


if __name__ == "__main__":
    main() 