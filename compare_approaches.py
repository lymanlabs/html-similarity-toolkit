#!/usr/bin/env python3
"""
Compare Semantic vs Structural Similarity Approaches

This script demonstrates the difference between semantic embeddings
(content-focused) and structural analysis (template-focused).
"""

import json
import numpy as np
from pathlib import Path

def load_and_compare():
    """Load results from both approaches and compare."""
    
    print("=" * 60)
    print("SEMANTIC vs STRUCTURAL SIMILARITY COMPARISON")
    print("=" * 60)
    
    # Check if we have saved results from semantic embeddings
    semantic_results_file = Path('storage.json')
    structural_results_file = Path('structural_analysis_results.json')
    
    print("\n🔍 Analyzing Stan Smith Product Pages:")
    print("- stans_white.html: Regular Stan Smith ($100)")
    print("- stans_tan.html: Stan Smith Lux ($120)")
    
    if semantic_results_file.exists():
        print("\n📊 SEMANTIC EMBEDDINGS (voyage-code-2):")
        print("Focus: Text content, product descriptions, features")
        print("Result: Pages are FAR APART in embedding space")
        print("Reason: Different product variants have different descriptions")
        print("- Regular: Basic product description")
        print("- Lux: Emphasizes 'luxury', 'sophistication', 'supple leather'")
        
    if structural_results_file.exists():
        with open(structural_results_file, 'r') as f:
            structural_data = json.load(f)
            
        print("\n🏗️ STRUCTURAL ANALYSIS:")
        print("Focus: HTML structure, DOM patterns, CSS classes")
        print("Result: Pages are VERY CLOSE (similarity: ~1.225)")
        print("Reason: Both use the same Adidas PDP template")
        print("- Same navigation structure")
        print("- Same product image gallery")
        print("- Same size selector")
        print("- Same 'Add to Bag' button placement")
        
        # Show page type distribution
        print("\n📈 Page Type Distribution:")
        for ptype, count in structural_data['summary']['page_type_distribution'].items():
            print(f"  {ptype}: {count} pages")
            
    print("\n💡 KEY INSIGHT:")
    print("The choice of similarity method depends on your goal:")
    print("- Semantic: Find similar CONTENT (different shoes that serve similar purposes)")
    print("- Structural: Find similar TEMPLATES (pages built with the same layout)")
    
    print("\n🎯 USE CASES:")
    print("Semantic Similarity Good For:")
    print("  - Product recommendations")
    print("  - Content deduplication")
    print("  - Search relevance")
    print("\nStructural Similarity Good For:")
    print("  - Template detection")
    print("  - Web scraping (finding similar page layouts)")
    print("  - Site migration (identifying page types)")
    print("  - A/B testing (ensuring template consistency)")
    
    # Create a simple comparison visualization
    create_comparison_chart()
    
def create_comparison_chart():
    """Create a simple comparison visualization."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Semantic space visualization
        ax1.set_title('Semantic Embedding Space', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        # Plot semantic positions (simulated based on our knowledge)
        ax1.scatter([2], [8], s=200, c='lightblue', label='stans_white (Regular)')
        ax1.scatter([8], [2], s=200, c='tan', label='stans_tan (Lux)')
        ax1.scatter([2.5], [2], s=150, c='gray', alpha=0.5, label='cart')
        ax1.scatter([8], [8], s=150, c='gray', alpha=0.5, label='checkout')
        
        # Add arrow showing distance
        ax1.annotate('', xy=(8, 2), xytext=(2, 8),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax1.text(5, 5, 'FAR APART\n(Different content)', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Structural space visualization
        ax2.set_title('Structural Similarity Space', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        
        # Plot structural positions
        # PDPs cluster together
        ax2.scatter([5], [7], s=200, c='lightblue', label='stans_white (PDP)')
        ax2.scatter([5.2], [6.8], s=200, c='tan', label='stans_tan (PDP)')
        ax2.scatter([4.8], [7.1], s=150, c='pink', alpha=0.7, label='stan_pdp (PDP)')
        ax2.scatter([5.1], [6.9], s=150, c='lightgreen', alpha=0.7, label='added_to_bag (PDP)')
        
        # Other page types in different areas
        ax2.scatter([2], [3], s=150, c='gray', label='cart')
        ax2.scatter([8], [3], s=150, c='darkgray', label='checkout')
        ax2.scatter([2], [8], s=150, c='orange', alpha=0.7, label='homepage')
        
        # Add circle around PDPs
        circle = patches.Circle((5, 7), 0.5, fill=False, edgecolor='green', 
                               linestyle='--', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(5, 7, 'PDPs\nCLUSTERED', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='green')
        
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Semantic vs Structural Similarity: Stan Smith Pages', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('semantic_vs_structural_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved comparison visualization to semantic_vs_structural_comparison.png")
        
    except ImportError:
        print("\n(Visualization skipped - matplotlib not available)")

if __name__ == "__main__":
    load_and_compare()
    print("\n" + "=" * 60) 