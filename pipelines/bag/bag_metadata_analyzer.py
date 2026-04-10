#!/usr/bin/env python3
"""
BAG Metadata Analyzer & Alignment Tool
Analyzes BAG file metadata and provides alignment corrections
"""

import os
import sys
import logging
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BagMetadataAnalyzer:
    """Analyzes BAG file metadata for alignment issues"""

    def __init__(self, bag_directory="bagfiles"):
        self.bag_directory = Path(bag_directory)
        self.metadata_analysis = {
            "files_analyzed": [],
            "crs_issues": [],
            "bounds_issues": [],
            "resolution_issues": [],
            "alignment_suggestions": [],
            "recommended_transforms": []
        }

    def analyze_all_bag_metadata(self):
        """Analyze metadata from all BAG files"""
        bag_files = list(self.bag_directory.glob("*.bag"))
        logger.info(f"Found {len(bag_files)} BAG files for metadata analysis")

        file_metadata = []

        for bag_file in bag_files:
            try:
                metadata = self.analyze_bag_metadata(bag_file)
                file_metadata.append(metadata)
                self.metadata_analysis["files_analyzed"].append(str(bag_file.name))
                logger.info(f"Analyzed metadata for: {bag_file.name}")
            except Exception as e:
                logger.error(f"Failed to analyze {bag_file.name}: {e}")

        # Cross-analyze metadata for alignment issues
        self.analyze_metadata_alignment(file_metadata)

        # Generate alignment recommendations
        self.generate_alignment_recommendations(file_metadata)

        self.save_analysis()

    def analyze_bag_metadata(self, bag_path):
        """Extract comprehensive metadata from a BAG file"""
        metadata = {
            "filename": bag_path.name,
            "path": str(bag_path),
            "basic_info": {},
            "spatial_info": {},
            "data_characteristics": {},
            "potential_issues": []
        }

        try:
            with rasterio.open(bag_path) as dataset:
                # Basic file information
                metadata["basic_info"] = {
                    "driver": dataset.driver,
                    "width": dataset.width,
                    "height": dataset.height,
                    "count": dataset.count,
                    "dtype": str(dataset.dtypes[0]) if dataset.dtypes else "unknown",
                    "crs": str(dataset.crs) if dataset.crs else None,
                    "transform": list(dataset.transform) if dataset.transform else None,
                    "bounds": list(dataset.bounds) if dataset.bounds else None,
                    "resolution": list(dataset.res) if dataset.res else None
                }

                # Spatial reference information
                metadata["spatial_info"] = {
                    "crs_wkt": dataset.crs.wkt if dataset.crs else None,
                    "crs_proj4": dataset.crs.to_proj4() if dataset.crs else None,
                    "is_geographic": dataset.crs.is_geographic if dataset.crs else None,
                    "is_projected": dataset.crs.is_projected if dataset.crs else None,
                    "datum": getattr(dataset.crs, 'datum', None),
                    "ellipsoid": getattr(dataset.crs, 'ellipsoid', None)
                }

                # Data characteristics
                elevation_data = dataset.read(1)  # Primary elevation band
                metadata["data_characteristics"] = {
                    "data_shape": elevation_data.shape,
                    "data_type": str(elevation_data.dtype),
                    "min_value": float(np.nanmin(elevation_data)) if not np.all(np.isnan(elevation_data)) else None,
                    "max_value": float(np.nanmax(elevation_data)) if not np.all(np.isnan(elevation_data)) else None,
                    "mean_value": float(np.nanmean(elevation_data)) if not np.all(np.isnan(elevation_data)) else None,
                    "nan_count": int(np.sum(np.isnan(elevation_data))),
                    "nan_percentage": float(np.sum(np.isnan(elevation_data)) / elevation_data.size * 100),
                    "valid_data_percentage": float((1 - np.sum(np.isnan(elevation_data)) / elevation_data.size) * 100)
                }

                # Check for common metadata issues
                issues = []

                # CRS issues
                if not dataset.crs:
                    issues.append("Missing coordinate reference system (CRS)")
                elif dataset.crs.is_geographic and abs(dataset.bounds.top) > 90:
                    issues.append("Geographic CRS with out-of-bounds latitudes")
                elif dataset.crs.is_projected and abs(dataset.bounds.left) > 20037508:  # Web Mercator bounds
                    issues.append("Projected CRS with potentially incorrect bounds")

                # Resolution issues
                if dataset.res:
                    x_res, y_res = dataset.res
                    if abs(abs(x_res) - abs(y_res)) / abs(x_res) > 0.01:  # More than 1% difference
                        issues.append(f"Uneven resolution: x={x_res}, y={y_res}")

                # Bounds issues
                if dataset.bounds:
                    left, bottom, right, top = dataset.bounds
                    if left >= right:
                        issues.append("Invalid bounds: left >= right")
                    if bottom >= top:
                        issues.append("Invalid bounds: bottom >= top")

                metadata["potential_issues"] = issues

        except Exception as e:
            logger.error(f"Error reading metadata from {bag_path.name}: {e}")
            metadata["error"] = str(e)

        return metadata

    def analyze_metadata_alignment(self, file_metadata):
        """Analyze metadata across files for alignment issues"""
        if len(file_metadata) < 2:
            logger.info("Need at least 2 files for alignment analysis")
            return

        # Collect CRS information
        crs_list = [m["basic_info"].get("crs") for m in file_metadata if m["basic_info"].get("crs")]
        unique_crs = set(crs_list)

        if len(unique_crs) > 1:
            self.metadata_analysis["crs_issues"].append({
                "issue": "Multiple CRS found",
                "crs_values": list(unique_crs),
                "files_affected": [m["filename"] for m in file_metadata],
                "severity": "high"
            })

        # Check bounds alignment
        bounds_list = []
        for m in file_metadata:
            bounds = m["basic_info"].get("bounds")
            if bounds:
                bounds_list.append((m["filename"], bounds))

        if len(bounds_list) > 1:
            # Check for overlapping or adjacent tiles
            self.analyze_bounds_alignment(bounds_list)

        # Check resolution consistency
        resolutions = []
        for m in file_metadata:
            res = m["basic_info"].get("resolution")
            if res:
                resolutions.append((m["filename"], res))

        if len(resolutions) > 1:
            self.analyze_resolution_consistency(resolutions)

    def analyze_bounds_alignment(self, bounds_list):
        """Analyze if bounds are properly aligned"""
        issues = []

        # Check for gaps or overlaps
        sorted_bounds = sorted(bounds_list, key=lambda x: x[1][0])  # Sort by left bound

        for i in range(len(sorted_bounds) - 1):
            current_file, current_bounds = sorted_bounds[i]
            next_file, next_bounds = sorted_bounds[i + 1]

            current_right = current_bounds[2]  # right
            next_left = next_bounds[0]       # left

            # Check for gap (with tolerance for floating point)
            gap = next_left - current_right
            if gap > 1e-6:  # Significant gap
                issues.append({
                    "type": "gap",
                    "files": [current_file, next_file],
                    "gap_distance": gap,
                    "location": f"between {current_right} and {next_left}"
                })
            elif gap < -1e-6:  # Overlap
                issues.append({
                    "type": "overlap",
                    "files": [current_file, next_file],
                    "overlap_distance": abs(gap),
                    "location": f"between {current_right} and {next_left}"
                })

        if issues:
            self.metadata_analysis["bounds_issues"].extend(issues)

    def analyze_resolution_consistency(self, resolutions):
        """Check if resolutions are consistent across files"""
        resolutions_only = [res[1] for res in resolutions]

        # Check if all resolutions are the same
        first_res = resolutions_only[0]
        inconsistent_files = []

        for filename, res in resolutions:
            if not np.allclose(res, first_res, rtol=1e-6):
                inconsistent_files.append((filename, res))

        if inconsistent_files:
            self.metadata_analysis["resolution_issues"].append({
                "reference_resolution": first_res,
                "inconsistent_files": inconsistent_files,
                "severity": "medium"
            })

    def generate_alignment_recommendations(self, file_metadata):
        """Generate recommendations for fixing alignment issues"""
        recommendations = []

        # CRS alignment recommendations
        if self.metadata_analysis["crs_issues"]:
            crs_issue = self.metadata_analysis["crs_issues"][0]
            recommendations.append({
                "issue_type": "crs_mismatch",
                "description": f"Multiple CRS found: {crs_issue['crs_values']}",
                "recommendation": "Reproject all files to a common CRS (e.g., EPSG:4326 for geographic or UTM zone appropriate for the area)",
                "priority": "high"
            })

        # Bounds alignment recommendations
        if self.metadata_analysis["bounds_issues"]:
            gap_issues = [i for i in self.metadata_analysis["bounds_issues"] if i["type"] == "gap"]
            overlap_issues = [i for i in self.metadata_analysis["bounds_issues"] if i["type"] == "overlap"]

            if gap_issues:
                recommendations.append({
                    "issue_type": "bounds_gaps",
                    "description": f"Found {len(gap_issues)} gaps between adjacent tiles",
                    "recommendation": "Adjust geotransforms to eliminate gaps, or use interpolation to fill missing data",
                    "priority": "medium"
                })

            if overlap_issues:
                recommendations.append({
                    "issue_type": "bounds_overlaps",
                    "description": f"Found {len(overlap_issues)} overlaps between tiles",
                    "recommendation": "Use mosaic tools to properly combine overlapping data, or adjust bounds to prevent overlap",
                    "priority": "medium"
                })

        # Resolution recommendations
        if self.metadata_analysis["resolution_issues"]:
            res_issue = self.metadata_analysis["resolution_issues"][0]
            recommendations.append({
                "issue_type": "resolution_inconsistency",
                "description": f"Inconsistent resolutions found across {len(res_issue['inconsistent_files'])} files",
                "recommendation": f"Resample all files to match reference resolution {res_issue['reference_resolution']}",
                "priority": "low"
            })

        # General recommendations
        if not recommendations:
            recommendations.append({
                "issue_type": "no_issues",
                "description": "No major alignment issues detected",
                "recommendation": "Files appear to be properly aligned",
                "priority": "info"
            })

        self.metadata_analysis["alignment_suggestions"] = recommendations

        # Generate specific transform recommendations
        self.generate_transform_recommendations(file_metadata)

    def generate_transform_recommendations(self, file_metadata):
        """Generate specific geotransform corrections"""
        transforms = []

        # If we have bounds issues, suggest specific corrections
        bounds_issues = self.metadata_analysis.get("bounds_issues", [])
        if bounds_issues:
            for issue in bounds_issues:
                if issue["type"] == "gap":
                    # Suggest shifting tiles to close gaps
                    gap_distance = issue["gap_distance"]
                    transforms.append({
                        "type": "translation",
                        "description": f"Shift tiles to close {gap_distance:.6f} unit gap",
                        "affected_files": issue["files"],
                        "transform_params": {"x_offset": gap_distance / 2, "y_offset": 0}
                    })

        self.metadata_analysis["recommended_transforms"] = transforms

    def save_analysis(self):
        """Save the metadata analysis results"""
        analysis_file = self.bag_directory / f"metadata_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(analysis_file, 'w') as f:
            json.dump(self.metadata_analysis, f, indent=2, default=str)

        logger.info(f"Metadata analysis saved to: {analysis_file}")

        # Print summary report
        self.print_summary_report()

    def print_summary_report(self):
        """Print a human-readable summary of the analysis"""
        print("\n" + "="*80)
        print("BAG METADATA ALIGNMENT ANALYSIS REPORT")
        print("="*80)

        print(f"📁 Files Analyzed: {len(self.metadata_analysis['files_analyzed'])}")
        print(f"🔍 CRS Issues: {len(self.metadata_analysis['crs_issues'])}")
        print(f"📐 Bounds Issues: {len(self.metadata_analysis['bounds_issues'])}")
        print(f"📏 Resolution Issues: {len(self.metadata_analysis['resolution_issues'])}")

        # Print recommendations
        recommendations = self.metadata_analysis.get("alignment_suggestions", [])
        if recommendations:
            print("\n💡 ALIGNMENT RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢", "info": "ℹ️"}.get(rec.get("priority", "info"), "ℹ️")
                print(f"   {i}. {priority_icon} {rec['description']}")
                print(f"      💡 {rec['recommendation']}")

        # Print specific transforms if available
        transforms = self.metadata_analysis.get("recommended_transforms", [])
        if transforms:
            print("\n🔧 SPECIFIC TRANSFORM RECOMMENDATIONS:")
            for i, transform in enumerate(transforms, 1):
                print(f"   {i}. {transform['description']}")
                if "transform_params" in transform:
                    params = transform["transform_params"]
                    print(f"      Parameters: {params}")

        print(f"\n💾 Detailed analysis saved to: metadata_analysis_*.json")
        print("="*80)

def main():
    """Main function to run metadata analysis"""
    print("🔍 BAG METADATA ALIGNMENT ANALYZER")
    print("=" * 50)
    print("🎯 Analyzing BAG file metadata for alignment issues")
    print("📊 Identifying CRS, bounds, and resolution mismatches")
    print("💡 Providing alignment correction recommendations")
    print()

    analyzer = BagMetadataAnalyzer()
    analyzer.analyze_all_bag_metadata()

if __name__ == "__main__":
    main()