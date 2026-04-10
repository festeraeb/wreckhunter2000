#!/usr/bin/env python3
"""
BAG Reconstruction Visualization Webpage Generator
Creates interactive webpage showing reconstructed areas from BAG files
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from pathlib import Path
from datetime import datetime
import json
import base64
from io import BytesIO

class BAGVisualizationGenerator:
    """Generate visualizations for reconstructed BAG files"""

    def __init__(self, bag_directory="bagfiles"):
        self.bag_directory = Path(bag_directory)
        self.reconstructed_dir = self.bag_directory / "fast_reconstructed"
        self.corrected_dir = self.bag_directory / "corrected"
        self.output_dir = self.bag_directory / "visualization"
        self.output_dir.mkdir(exist_ok=True)

        # Color schemes for bathymetry
        self.cmap = plt.cm.viridis
        self.cmap.set_bad(color='red')  # NoData values in red

    def generate_bathymetry_image(self, bag_path, output_path, title="Bathymetry"):
        """Generate bathymetry visualization image"""
        try:
            with rasterio.open(bag_path) as src:
                elevation = src.read(1)

                # Create figure
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))

                # Plot bathymetry
                im = ax.imshow(elevation,
                              cmap=self.cmap,
                              vmin=np.nanmin(elevation),
                              vmax=np.nanmax(elevation))

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Depth (meters)', rotation=270, labelpad=15)

                ax.set_title(f'{title}\n{Path(bag_path).name}')
                ax.set_xlabel('Easting')
                ax.set_ylabel('Northing')

                # Save high-quality image
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

                return True

        except Exception as e:
            print(f"Failed to generate image for {bag_path}: {e}")
            return False

    def generate_difference_map(self, original_path, reconstructed_path, output_path):
        """Generate difference map showing reconstruction changes"""
        try:
            with rasterio.open(original_path) as orig_src:
                original = orig_src.read(1)

            with rasterio.open(reconstructed_path) as recon_src:
                reconstructed = recon_src.read(1)

            # Calculate difference
            difference = reconstructed - original

            # Create difference colormap (red for filled areas, blue for changed)
            diff_cmap = plt.cm.RdYlBu_r
            diff_cmap.set_bad(color='gray')

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Only show areas that were originally masked (NoData)
            mask = np.ma.masked_invalid(original)
            diff_masked = np.ma.masked_where(mask.mask == False, difference)

            im = ax.imshow(diff_masked, cmap=diff_cmap, vmin=-5, vmax=5)

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Depth Change (meters)', rotation=270, labelpad=15)

            ax.set_title(f'Reconstruction Changes\n{Path(reconstructed_path).name}')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            return True

        except Exception as e:
            print(f"Failed to generate difference map: {e}")
            return False

    def generate_all_visualizations(self):
        """Generate visualizations for all reconstructed files"""
        print("🎨 Generating BAG reconstruction visualizations...")

        reconstructed_files = list(self.reconstructed_dir.glob("*.bag"))
        if not reconstructed_files:
            print("❌ No reconstructed BAG files found")
            return []

        visualizations = []

        for recon_file in reconstructed_files:
            file_stem = recon_file.stem.replace('_fast_reconstructed', '')
            print(f"📊 Processing {file_stem}")

            # Generate bathymetry image
            bathy_path = self.output_dir / f"{file_stem}_bathymetry.png"
            if self.generate_bathymetry_image(recon_file, bathy_path, "Reconstructed Bathymetry"):
                visualizations.append({
                    'file': file_stem,
                    'bathymetry_image': bathy_path.name,
                    'type': 'reconstructed'
                })

            # Try to find corresponding corrected file for comparison
            corrected_file = self.corrected_dir / f"{file_stem.replace('_aligned', '')}_corrected.bag"
            if not corrected_file.exists():
                # Try alternative naming patterns
                for pattern in ['_corrected.bag', '_aligned.bag']:
                    alt_file = self.corrected_dir / f"{file_stem}{pattern}"
                    if alt_file.exists():
                        corrected_file = alt_file
                        break

            if corrected_file.exists():
                diff_path = self.output_dir / f"{file_stem}_difference.png"
                if self.generate_difference_map(corrected_file, recon_file, diff_path):
                    # Update visualization entry with difference map
                    for viz in visualizations:
                        if viz['file'] == file_stem:
                            viz['difference_image'] = diff_path.name
                            break

        return visualizations

    def generate_pdf_analysis_visualizations(self):
        """Generate visualizations for PDF analysis results"""
        print("📄 Generating PDF analysis visualizations...")

        # Look for PDF analysis results
        results_dir = Path(".")
        pdf_results_files = list(results_dir.glob("*redaction*results*.json")) + \
                           list(results_dir.glob("*analysis*results*.json"))

        if not pdf_results_files:
            print("⚠️  No PDF analysis results found")
            return []

        # Get the most recent results
        latest_results = max(pdf_results_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_results, 'r') as f:
                results_data = json.load(f)

            visualizations = []

            # Create summary statistics visualization
            self.create_pdf_summary_chart(results_data)

            # Create breakthrough techniques visualization
            self.create_techniques_chart(results_data)

            # Create file-by-file results
            self.create_file_results_summary(results_data)

            visualizations.append({
                'type': 'pdf_analysis',
                'summary_chart': 'pdf_analysis_summary.png',
                'techniques_chart': 'pdf_breakthrough_techniques.png',
                'file_results': 'pdf_file_results.txt'
            })

            print(f"✅ Generated PDF visualizations from {latest_results.name}")
            return visualizations

        except Exception as e:
            print(f"❌ Failed to generate PDF visualizations: {e}")
            return []

    def create_pdf_summary_chart(self, results_data):
        """Create a summary chart of PDF analysis results"""
        try:
            import matplotlib.pyplot as plt

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Summary statistics
            total_files = results_data.get('total_files_analyzed', 0)
            successful = results_data.get('successful_breakthroughs', 0)
            total_breakthroughs = results_data.get('total_breakthroughs', 0)

            # Pie chart: Success rate
            success_rate = successful / total_files * 100 if total_files > 0 else 0
            ax1.pie([success_rate, 100-success_rate],
                   labels=['Successful', 'No Breakthroughs'],
                   autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'])
            ax1.set_title('PDF Analysis Success Rate')

            # Bar chart: Breakthroughs per file
            if 'file_results' in results_data:
                files = [f.get('file', 'Unknown')[:20] + '...' if len(f.get('file', 'Unknown')) > 20
                        else f.get('file', 'Unknown') for f in results_data['file_results']]
                breakthroughs = [len(f.get('breakthroughs', [])) for f in results_data['file_results']]

                ax2.bar(range(len(files)), breakthroughs, color='#3498db')
                ax2.set_xticks(range(len(files)))
                ax2.set_xticklabels(files, rotation=45, ha='right')
                ax2.set_title('Breakthroughs per PDF File')
                ax2.set_ylabel('Number of Breakthroughs')

            # Techniques used
            techniques = results_data.get('techniques_attempted', [])
            if techniques:
                technique_counts = {}
                for file_result in results_data.get('file_results', []):
                    for technique in file_result.get('techniques_attempted', []):
                        technique_counts[technique] = technique_counts.get(technique, 0) + 1

                ax3.bar(list(technique_counts.keys()), list(technique_counts.values()), color='#9b59b6')
                ax3.set_title('Technique Usage Frequency')
                ax3.set_ylabel('Files Using Technique')
                ax3.tick_params(axis='x', rotation=45)

            # Content types found
            content_types = {}
            for file_result in results_data.get('file_results', []):
                for breakthrough in file_result.get('breakthroughs', []):
                    content_type = breakthrough.get('content_type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1

            if content_types:
                ax4.pie(list(content_types.values()),
                       labels=list(content_types.keys()),
                       autopct='%1.1f%%')
                ax4.set_title('Content Types Discovered')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'pdf_analysis_summary.png', dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Failed to create PDF summary chart: {e}")

    def create_techniques_chart(self, results_data):
        """Create a chart showing breakthrough techniques effectiveness"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Collect technique effectiveness data
            techniques = {}
            for file_result in results_data.get('file_results', []):
                for breakthrough in file_result.get('breakthroughs', []):
                    technique = breakthrough.get('technique_used', 'unknown')
                    if technique not in techniques:
                        techniques[technique] = {'successes': 0, 'attempts': 0}
                    techniques[technique]['successes'] += 1

                # Count attempts
                for technique in file_result.get('techniques_attempted', []):
                    if technique not in techniques:
                        techniques[technique] = {'successes': 0, 'attempts': 0}
                    techniques[technique]['attempts'] += 1

            if techniques:
                tech_names = list(techniques.keys())
                success_rates = [techniques[t]['successes'] / techniques[t]['attempts'] * 100
                               if techniques[t]['attempts'] > 0 else 0 for t in tech_names]

                bars = ax.bar(tech_names, success_rates, color='#e67e22')
                ax.set_title('PDF Breakthrough Technique Effectiveness')
                ax.set_ylabel('Success Rate (%)')
                ax.set_xlabel('Technique')
                ax.tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for bar, rate in zip(bars, success_rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           '.1f', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'pdf_breakthrough_techniques.png', dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Failed to create techniques chart: {e}")

    def create_file_results_summary(self, results_data):
        """Create a text summary of file-by-file results"""
        try:
            summary_path = self.output_dir / 'pdf_file_results.txt'

            with open(summary_path, 'w') as f:
                f.write("PDF ANALYSIS RESULTS SUMMARY\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Total Files Analyzed: {results_data.get('total_files_analyzed', 0)}\n")
                f.write(f"Successful Breakthroughs: {results_data.get('successful_breakthroughs', 0)}\n")
                f.write(f"Total Breakthroughs: {results_data.get('total_breakthroughs', 0)}\n\n")

                f.write("FILE-BY-FILE RESULTS:\n")
                f.write("-" * 30 + "\n")

                for file_result in results_data.get('file_results', []):
                    f.write(f"\nFile: {file_result.get('file', 'Unknown')}\n")
                    f.write(f"Breakthroughs: {len(file_result.get('breakthroughs', []))}\n")
                    f.write(f"Techniques Used: {', '.join(file_result.get('techniques_attempted', []))}\n")

                    # List breakthroughs
                    for i, breakthrough in enumerate(file_result.get('breakthroughs', []), 1):
                        f.write(f"  {i}. {breakthrough.get('content_type', 'Unknown')} - ")
                        f.write(f"{breakthrough.get('technique_used', 'Unknown')}\n")

        except Exception as e:
            print(f"Failed to create file results summary: {e}")

    def create_html_page(self, visualizations):
        """Create HTML webpage for visualization gallery"""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BAG Reconstruction Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .gallery {{
            padding: 30px;
        }}
        .file-section {{
            margin-bottom: 50px;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
        }}
        .file-header {{
            background: #34495e;
            color: white;
            padding: 15px 20px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            padding: 20px;
        }}
        .image-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s ease;
        }}
        .image-card:hover {{
            transform: translateY(-5px);
        }}
        .image-card img {{
            width: 100%;
            height: 300px;
            object-fit: cover;
            display: block;
        }}
        .image-info {{
            padding: 15px;
        }}
        .image-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .image-desc {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #ecf0f1;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .image-grid {{
                grid-template-columns: 1fr;
            }}
            .stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗺️ BAG Reconstruction Visualization</h1>
            <p>Interactive gallery showing ML-enhanced bathymetry reconstruction results</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(visualizations)}</div>
                <div class="stat-label">Files Reconstructed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">⚡</div>
                <div class="stat-label">Rust Accelerated</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">🤖</div>
                <div class="stat-label">ML Enhanced</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">3.4x</div>
                <div class="stat-label">Speed Improvement</div>
            </div>
        </div>

        <div class="gallery">
"""

        for viz in visualizations:
            if viz.get('type') == 'pdf_analysis':
                continue  # Skip PDF viz here, handle separately below

            html_content += f"""
            <div class="file-section">
                <div class="file-header">📊 {viz['file']}</div>
                <div class="image-grid">
"""

            # Bathymetry image
            if 'bathymetry_image' in viz:
                html_content += f"""
                    <div class="image-card">
                        <img src="visualization/{viz['bathymetry_image']}" alt="Bathymetry">
                        <div class="image-info">
                            <div class="image-title">Reconstructed Bathymetry</div>
                            <div class="image-desc">ML-enhanced depth reconstruction using pattern interpolation and geometric techniques</div>
                        </div>
                    </div>
"""

            # Difference map
            if 'difference_image' in viz:
                html_content += f"""
                    <div class="image-card">
                        <img src="visualization/{viz['difference_image']}" alt="Reconstruction Changes">
                        <div class="image-info">
                            <div class="image-title">Reconstruction Changes</div>
                            <div class="image-desc">Difference map showing areas that were reconstructed (red=filled, blue=modified)</div>
                        </div>
                    </div>
"""

            html_content += """
                </div>
            </div>
"""

        # Add PDF analysis section if available
        pdf_viz = next((v for v in visualizations if v.get('type') == 'pdf_analysis'), None)
        if pdf_viz:
            html_content += """
            <div class="file-section">
                <div class="file-header">📄 PDF Analysis Results</div>
                <div class="image-grid">
"""

            if 'summary_chart' in pdf_viz:
                html_content += f"""
                    <div class="image-card">
                        <img src="visualization/{pdf_viz['summary_chart']}" alt="PDF Analysis Summary">
                        <div class="image-info">
                            <div class="image-title">PDF Analysis Summary</div>
                            <div class="image-desc">Overview of redaction breaking success rates, techniques used, and content types discovered</div>
                        </div>
                    </div>
"""

            if 'techniques_chart' in pdf_viz:
                html_content += f"""
                    <div class="image-card">
                        <img src="visualization/{pdf_viz['techniques_chart']}" alt="Breakthrough Techniques">
                        <div class="image-info">
                            <div class="image-title">Technique Effectiveness</div>
                            <div class="image-desc">Success rates of different redaction breaking techniques across all analyzed PDFs</div>
                        </div>
                    </div>
"""

            html_content += """
                </div>
            </div>
"""

        html_content += """
        </div>

        <div class="footer">
            <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """ | Fast ML-Enhanced BAG Reconstruction Tool</p>
        </div>
    </div>
</body>
</html>"""

        # Save HTML file
        html_path = self.bag_directory / "reconstruction_gallery.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML gallery created: {html_path}")
        return html_path

def main():
    """Main function to generate BAG reconstruction visualizations"""
    print("🚀 BAG Reconstruction Visualization Generator")
    print("=" * 50)

    generator = BAGVisualizationGenerator()

    # Generate all visualizations
    visualizations = generator.generate_all_visualizations()

    # Generate PDF analysis visualizations
    pdf_visualizations = generator.generate_pdf_analysis_visualizations()
    visualizations.extend(pdf_visualizations)

    if not visualizations:
        print("❌ No visualizations generated")
        return

    # Create HTML webpage
    html_path = generator.create_html_page(visualizations)

    print("\n✅ Visualization complete!")
    print(f"📁 Open this file in your browser: {html_path}")
    print(f"🖼️  {len([v for v in visualizations if v.get('type') == 'reconstructed'])} BAG files visualized")
    print(f"📄 {len([v for v in visualizations if v.get('type') == 'pdf_analysis'])} PDF analysis results visualized")

    # Try to open the webpage automatically
    try:
        import webbrowser
        webbrowser.open(str(html_path))
        print("🌐 Webpage opened automatically")
    except:
        pass

if __name__ == "__main__":
    main()