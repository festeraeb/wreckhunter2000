#!/usr/bin/env python3
"""
BAG File Alignment Correction Tool
Fixes metadata mismatches and aligns BAG files properly
"""

import os
import sys
import logging
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pathlib import Path
from datetime import datetime
import json
import shutil
from rasterio.crs import CRS
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BagAlignmentCorrector:
    """Corrects BAG file alignment and metadata issues"""

    def __init__(self, bag_directory="bagfiles"):
        self.bag_directory = Path(bag_directory)
        self.corrected_directory = self.bag_directory / "corrected"
        self.corrected_directory.mkdir(exist_ok=True)

        self.correction_log = {
            "corrections_applied": [],
            "files_processed": [],
            "alignment_improvements": [],
            "metadata_fixes": []
        }

    def load_metadata_analysis(self):
        """Load the metadata analysis results"""
        analysis_files = list(self.bag_directory.glob("metadata_analysis_*.json"))
        if not analysis_files:
            logger.error("No metadata analysis file found. Run metadata analyzer first.")
            return None

        # Get the most recent analysis
        analysis_file = max(analysis_files, key=lambda x: x.stat().st_mtime)

        with open(analysis_file, 'r') as f:
            return json.load(f)

    def apply_alignment_corrections(self):
        """Apply all identified alignment corrections"""
        logger.info("🔧 Starting BAG file alignment corrections")

        # Load metadata analysis
        analysis = self.load_metadata_analysis()
        if not analysis:
            return

        # Get file metadata for processing
        file_metadata = self.get_detailed_file_metadata(analysis["files_analyzed"])

        # Apply corrections based on analysis
        self.correct_resolution_inconsistencies(file_metadata, analysis)
        self.correct_bounds_overlaps(file_metadata, analysis)

        # Save correction log
        self.save_correction_log()

        logger.info("✅ Alignment corrections completed")

    def get_detailed_file_metadata(self, filenames):
        """Get detailed metadata for each file"""
        file_metadata = {}

        for filename in filenames:
            bag_path = self.bag_directory / filename
            if bag_path.exists():
                try:
                    with rasterio.open(bag_path) as dataset:
                        file_metadata[filename] = {
                            "path": bag_path,
                            "width": dataset.width,
                            "height": dataset.height,
                            "bounds": list(dataset.bounds),
                            "resolution": list(dataset.res),
                            "transform": list(dataset.transform),
                            "crs": str(dataset.crs) if dataset.crs else None
                        }
                except Exception as e:
                    logger.error(f"Failed to read metadata for {filename}: {e}")

        return file_metadata

    def correct_resolution_inconsistencies(self, file_metadata, analysis):
        """Correct resolution inconsistencies by resampling"""
        resolution_issues = analysis.get("resolution_issues", [])
        if not resolution_issues:
            return

        issue = resolution_issues[0]  # Take the first (and likely only) issue
        reference_resolution = issue["reference_resolution"]
        inconsistent_files = issue["inconsistent_files"]

        logger.info(f"🔧 Correcting resolution inconsistencies to {reference_resolution}")

        for filename, current_res in inconsistent_files:
            if filename in file_metadata:
                try:
                    self.resample_file(filename, file_metadata[filename], reference_resolution)
                    self.correction_log["corrections_applied"].append({
                        "type": "resolution_resampling",
                        "file": filename,
                        "from_resolution": current_res,
                        "to_resolution": reference_resolution
                    })
                except Exception as e:
                    logger.error(f"Failed to resample {filename}: {e}")

    def resample_file(self, filename, metadata, target_resolution):
        """Resample a file to target resolution"""
        input_path = metadata["path"]
        filename_path = Path(filename)
        output_path = self.corrected_directory / f"{filename_path.stem}_corrected.bag"

        target_x_res, target_y_res = target_resolution

        with rasterio.open(input_path) as src:
            # Calculate new dimensions
            new_width = int(src.width * src.res[0] / target_x_res)
            new_height = int(src.height * abs(src.res[1]) / abs(target_y_res))

            # Create new transform
            new_transform = from_bounds(
                src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top,
                new_width, new_height
            )

            # Create output profile
            profile = src.profile.copy()
            profile.update({
                'width': new_width,
                'height': new_height,
                'transform': new_transform,
                'crs': src.crs
            })

            # Resample and write
            with rasterio.open(output_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=new_transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear
                    )

        logger.info(f"Resampled {filename} from {metadata['resolution']} to {target_resolution}")

    def correct_bounds_overlaps(self, file_metadata, analysis):
        """Correct bounds overlaps by adjusting geotransforms"""
        bounds_issues = analysis.get("bounds_issues", [])
        if not bounds_issues:
            return

        logger.info("🔧 Correcting bounds overlaps")

        # Group files by their spatial relationships
        overlap_groups = self.group_overlapping_files(bounds_issues, file_metadata)

        for group in overlap_groups:
            try:
                self.adjust_group_alignment(group, file_metadata)
            except Exception as e:
                logger.error(f"Failed to adjust group alignment: {e}")

    def group_overlapping_files(self, bounds_issues, file_metadata):
        """Group files that have overlapping bounds"""
        groups = []

        # Simple grouping based on overlap issues
        processed_files = set()

        for issue in bounds_issues:
            if issue["type"] == "overlap":
                files = issue["files"]
                overlap_distance = issue["overlap_distance"]

                # Only process significant overlaps
                if overlap_distance > 100:  # More than 100 units overlap
                    group = []
                    for filename in files:
                        if filename not in processed_files and filename in file_metadata:
                            group.append(filename)
                            processed_files.add(filename)

                    if len(group) > 1:
                        groups.append({
                            "files": group,
                            "overlap_distance": overlap_distance,
                            "issue": issue
                        })

        return groups

    def adjust_group_alignment(self, group, file_metadata):
        """Adjust the alignment of a group of overlapping files"""
        files = group["files"]
        overlap_distance = group["overlap_distance"]

        logger.info(f"Adjusting alignment for group: {files} (overlap: {overlap_distance})")

        # Strategy: Shift the second file in each pair to eliminate overlap
        # This is a simplified approach - in practice, you'd want more sophisticated mosaicking

        for i in range(len(files) - 1):
            file1 = files[i]
            file2 = files[i + 1]

            if file1 in file_metadata and file2 in file_metadata:
                # Calculate shift needed (half the overlap distance)
                shift_distance = overlap_distance / 2

                # Adjust the transform of file2
                self.shift_file_transform(file2, file_metadata[file2], shift_distance)

                self.correction_log["corrections_applied"].append({
                    "type": "bounds_adjustment",
                    "files": [file1, file2],
                    "shift_applied": shift_distance,
                    "overlap_corrected": overlap_distance
                })

    def shift_file_transform(self, filename, metadata, shift_distance):
        """Shift a file's geotransform to adjust positioning"""
        input_path = metadata["path"]
        filename_path = Path(filename)
        output_path = self.corrected_directory / f"{filename_path.stem}_aligned.bag"

        with rasterio.open(input_path) as src:
            # Create new transform with x-shift
            old_transform = src.transform
            new_transform = rasterio.transform.from_bounds(
                src.bounds.left + shift_distance,  # Shift left bound
                src.bounds.bottom,
                src.bounds.right + shift_distance,  # Shift right bound
                src.bounds.top,
                src.width,
                src.height
            )

            # Copy the file with new transform
            profile = src.profile.copy()
            profile.update({'transform': new_transform})

            with rasterio.open(output_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)

        logger.info(f"Shifted {filename} by {shift_distance} units")

    def fix_corner_points_consistency(self):
        """Fix the GDAL warnings about corner points consistency"""
        logger.info("🔧 Fixing corner points consistency issues")

        bag_files = list(self.bag_directory.glob("*.bag"))

        for bag_path in bag_files:
            try:
                # Read current metadata
                with rasterio.open(bag_path) as src:
                    current_bounds = src.bounds
                    current_transform = src.transform
                    resolution = src.res

                    # Calculate expected bounds from transform and dimensions
                    expected_right = current_transform.c + current_transform.a * src.width
                    expected_bottom = current_transform.f + current_transform.e * src.height

                    # Check if bounds are consistent
                    bounds_consistent = (
                        abs(current_bounds.right - expected_right) < 1e-6 and
                        abs(current_bounds.bottom - expected_bottom) < 1e-6
                    )

                    if not bounds_consistent:
                        logger.info(f"Fixing bounds consistency for {bag_path.name}")

                        # Create corrected bounds
                        corrected_bounds = (
                            current_bounds.left,
                            expected_bottom,
                            expected_right,
                            current_bounds.top
                        )

                        # Create new transform from corrected bounds
                        corrected_transform = from_bounds(*corrected_bounds, src.width, src.height)

                        # Write corrected file
                        output_path = self.corrected_directory / f"{bag_path.stem}_bounds_fixed.bag"

                        profile = src.profile.copy()
                        profile.update({'transform': corrected_transform})

                        with rasterio.open(output_path, 'w', **profile) as dst:
                            for i in range(1, src.count + 1):
                                dst.write(src.read(i), i)

                        self.correction_log["metadata_fixes"].append({
                            "type": "bounds_consistency",
                            "file": bag_path.name,
                            "original_bounds": list(current_bounds),
                            "corrected_bounds": list(corrected_bounds)
                        })

            except Exception as e:
                logger.error(f"Failed to fix bounds for {bag_path.name}: {e}")

    def create_mosaic_preview(self):
        """Create a preview mosaic of all corrected files"""
        try:
            corrected_files = list(self.corrected_directory.glob("*_corrected.bag")) + \
                            list(self.corrected_directory.glob("*_aligned.bag")) + \
                            list(self.corrected_directory.glob("*_bounds_fixed.bag"))

            if len(corrected_files) < 2:
                logger.info("Need at least 2 corrected files for mosaic preview")
                return

            logger.info(f"Creating mosaic preview from {len(corrected_files)} files")

            src_datasets = []
            try:
                # Open corrected rasters for merge.
                for p in corrected_files:
                    src_datasets.append(rasterio.open(p))

                # Keep only sources matching the first CRS to avoid silent misalignment.
                base_crs = src_datasets[0].crs
                compatible = [ds for ds in src_datasets if ds.crs == base_crs]
                skipped = len(src_datasets) - len(compatible)
                if len(compatible) < 2:
                    logger.warning("Not enough CRS-compatible files for mosaic preview")
                    return

                # Merge first band into a single georeferenced mosaic.
                mosaic_arr, mosaic_transform = merge(compatible, indexes=1)
                mosaic = mosaic_arr[0]

                out_tif = self.corrected_directory / "mosaic_preview.tif"
                profile = compatible[0].profile.copy()
                profile.update({
                    'driver': 'GTiff',
                    'height': mosaic.shape[0],
                    'width': mosaic.shape[1],
                    'count': 1,
                    'transform': mosaic_transform,
                    'crs': base_crs,
                    'compress': 'deflate',
                    'tiled': True,
                    'blockxsize': 256,
                    'blockysize': 256,
                })

                with rasterio.open(out_tif, 'w', **profile) as dst:
                    dst.write(mosaic, 1)

                out_png = self.corrected_directory / "mosaic_preview.png"
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                    finite = mosaic[np.isfinite(mosaic)]
                    if finite.size > 0:
                        vmin = np.nanpercentile(finite, 2)
                        vmax = np.nanpercentile(finite, 98)
                    else:
                        vmin, vmax = 0.0, 1.0

                    plt.figure(figsize=(12, 8))
                    plt.imshow(mosaic, cmap='terrain', vmin=vmin, vmax=vmax, origin='upper')
                    plt.title('Corrected BAG Mosaic Preview')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(out_png, dpi=180)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Failed to write PNG mosaic preview: {e}")

                self.correction_log["alignment_improvements"].append({
                    "type": "mosaic_preview_generated",
                    "files_available": len(corrected_files),
                    "files_used": len(compatible),
                    "files_skipped_crs_mismatch": skipped,
                    "mosaic_tif": str(out_tif),
                    "mosaic_png": str(out_png)
                })

                logger.info(f"Mosaic preview written: {out_tif}")
                if out_png.exists():
                    logger.info(f"Mosaic quicklook written: {out_png}")
            finally:
                for ds in src_datasets:
                    try:
                        ds.close()
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Failed to create mosaic preview: {e}")

    def save_correction_log(self):
        """Save the correction log"""
        log_file = self.bag_directory / f"alignment_corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(log_file, 'w') as f:
            json.dump(self.correction_log, f, indent=2, default=str)

        logger.info(f"Correction log saved to: {log_file}")

        # Print summary
        print("\n" + "="*70)
        print("BAG ALIGNMENT CORRECTION SUMMARY")
        print("="*70)
        print(f"📁 Files Processed: {len(self.correction_log['files_processed'])}")
        print(f"🔧 Corrections Applied: {len(self.correction_log['corrections_applied'])}")
        print(f"📐 Metadata Fixes: {len(self.correction_log['metadata_fixes'])}")
        print(f"🎯 Alignment Improvements: {len(self.correction_log['alignment_improvements'])}")

        if self.correction_log['corrections_applied']:
            print("\n✅ CORRECTIONS APPLIED:")
            for correction in self.correction_log['corrections_applied']:
                print(f"   • {correction['type']}: {correction.get('file', 'multiple files')}")

        print(f"\n💾 Corrected files saved in: {self.corrected_directory}")
        print(f"📋 Detailed log: {log_file}")
        print("="*70)

def main():
    """Main function to run alignment corrections"""
    print("🔧 BAG FILE ALIGNMENT CORRECTOR")
    print("=" * 50)
    print("🎯 Fixing metadata mismatches and alignment issues")
    print("📐 Correcting bounds overlaps and resolution inconsistencies")
    print("🔄 Creating properly aligned BAG files")
    print()

    corrector = BagAlignmentCorrector()

    # Apply corrections
    corrector.apply_alignment_corrections()

    # Fix corner points consistency
    corrector.fix_corner_points_consistency()

    # Create mosaic preview info
    corrector.create_mosaic_preview()

if __name__ == "__main__":
    main()