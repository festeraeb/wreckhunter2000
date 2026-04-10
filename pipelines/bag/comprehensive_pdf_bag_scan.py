"""
Comprehensive PDF-BAG Cross-Reference Scanner
Extracts coordinates from ALL PDFs and checks against ALL BAG files
Reports both visible wrecks and suspected scrubbed (masked) wrecks

All measurements in FEET and NAUTICAL MILES
"""
import os
import re
import json
import fitz
import rasterio
import numpy as np
from scipy import ndimage
from pyproj import Transformer
from pathlib import Path
from datetime import datetime

# Conversion constants
M_TO_FT = 3.28084
NM_TO_FT = 6076.12

class ComprehensivePDFBAGScanner:
    def __init__(self):
        self.pdf_dir = Path("PDFS Original")
        self.bag_dirs = [
            Path("development_and_tools/bagfiles"),
            Path("Lake Erie Bag Files")
        ]
        
        # Known wrecks for validation
        self.known_wrecks = {
            "Elva": {"lat": 45.849306, "lon": -84.613028, "depth_ft": 40, "length_ft": 135, "status": "MASKED"},
            "Elva Barge": {"lat": 45.849194, "lon": -84.612333, "depth_ft": 45, "length_ft": 100, "status": "MASKED"},
            "Cedarville": {"lat": 45.8175, "lon": -84.6058, "depth_ft": 105, "length_ft": 588, "status": "PUBLISHED"},
            "Nordmeer": {"lat": 45.8181, "lon": -84.6047, "depth_ft": 95, "length_ft": 469, "status": "PUBLISHED"},
        }
        
        self.results = {
            "scan_date": datetime.now().isoformat(),
            "unit_system": "Imperial (feet, nautical miles)",
            "pdfs_processed": [],
            "coordinates_extracted": [],
            "wreck_signatures": [],
            "scrubbed_locations": [],
            "high_texture": [],
            "known_wreck_matches": [],
        }
        
    def find_bag_files(self):
        """Find all BAG files"""
        bag_files = []
        for d in self.bag_dirs:
            if d.exists():
                bag_files.extend(list(d.glob("*.bag")))
        return bag_files
    
    def extract_coordinates_from_pdf(self, pdf_path):
        """Extract all coordinates from a PDF"""
        coords = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # DMS format: 45° 49' 22.5" N
                lat_pattern = r"(\d{2})\s*[°]\s*(\d{2})\s*['\u2032]\s*(\d{2}\.?\d*)\s*[\"\u2033]?\s*([NS])"
                lon_pattern = r"(\d{2,3})\s*[°]\s*(\d{2})\s*['\u2032]\s*(\d{2}\.?\d*)\s*[\"\u2033]?\s*([EW])"
                
                lat_matches = re.findall(lat_pattern, text)
                lon_matches = re.findall(lon_pattern, text)
                
                for lat_m, lon_m in zip(lat_matches, lon_matches):
                    lat = float(lat_m[0]) + float(lat_m[1])/60 + float(lat_m[2])/3600
                    if lat_m[3] == 'S': lat = -lat
                    
                    lon = float(lon_m[0]) + float(lon_m[1])/60 + float(lon_m[2])/3600
                    if lon_m[3] == 'W': lon = -lon
                    
                    coords.append({
                        "lat": lat, "lon": lon, 
                        "page": page_num + 1,
                        "source": "DMS",
                        "pdf": pdf_path.name
                    })
            
            doc.close()
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # Deduplicate
        unique = []
        for c in coords:
            is_dup = any(abs(c['lat']-u['lat']) < 0.0001 and abs(c['lon']-u['lon']) < 0.0001 for u in unique)
            if not is_dup:
                unique.append(c)
        
        return unique
    
    def find_bag_for_coordinates(self, lat, lon, bag_files):
        """Find the best BAG file covering the given coordinates"""
        for bag_path in bag_files:
            try:
                with rasterio.open(bag_path) as src:
                    bounds = src.bounds
                    crs = src.crs
                    
                    # Transform bounds to lat/lon
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
                    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
                    
                    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                        return bag_path
            except:
                continue
        return None
    
    def analyze_location(self, bag_path, lat, lon):
        """Analyze a location for wreck signatures"""
        try:
            with rasterio.open(bag_path) as src:
                elevation = src.read(1)
                transform = src.transform
                crs = src.crs
                pixel_size_m = abs(transform[0])
                pixel_size_ft = pixel_size_m * M_TO_FT
                
                transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                
                col = int((x - transform[2]) / transform[0])
                row = int((y - transform[5]) / transform[4])
                
                h, w = elevation.shape
                if col < 0 or col >= w or row < 0 or row >= h:
                    return None
                
                # 300 ft search radius
                search_px = int(300 / pixel_size_ft)
                r1, r2 = max(0, row-search_px), min(h, row+search_px)
                c1, c2 = max(0, col-search_px), min(w, col+search_px)
                
                local = elevation[r1:r2, c1:c2]
                valid_mask = (local > -1000) & (local < 1000)
                valid = local[valid_mask]
                
                if len(valid) < 100:
                    return None
                
                # Metrics in FEET
                depth_ft = abs(np.mean(valid)) * M_TO_FT
                std_ft = np.std(valid) * M_TO_FT
                range_ft = (np.max(valid) - np.min(valid)) * M_TO_FT
                
                # Gradient
                local_clean = np.where(valid_mask, local, np.nan)
                gx = ndimage.sobel(local_clean, axis=1)
                gy = ndimage.sobel(local_clean, axis=0)
                gradient = np.sqrt(gx**2 + gy**2)
                grad_mean_ft = np.nanmean(gradient) * M_TO_FT
                grad_max_ft = np.nanmax(gradient) * M_TO_FT
                
                # Shape analysis
                below_mean = local_clean < (np.nanmean(local_clean) - np.nanstd(local_clean))
                if np.any(below_mean):
                    rows_extent = np.sum(np.any(below_mean, axis=1)) * pixel_size_ft
                    cols_extent = np.sum(np.any(below_mean, axis=0)) * pixel_size_ft
                    aspect = max(rows_extent, cols_extent) / max(min(rows_extent, cols_extent), 1)
                else:
                    rows_extent, cols_extent, aspect = 0, 0, 1
                
                # Classification
                if std_ft < 1.0 and grad_mean_ft < 0.15:
                    signature = "SCRUBBED"
                elif grad_max_ft > 3.0 and aspect > 1.8:
                    signature = "WRECK_SIGNATURE"
                elif grad_mean_ft > 0.8:
                    signature = "HIGH_TEXTURE"
                else:
                    signature = "NORMAL"
                
                return {
                    "depth_ft": round(depth_ft, 1),
                    "std_ft": round(std_ft, 2),
                    "range_ft": round(range_ft, 1),
                    "grad_mean_ft": round(grad_mean_ft, 3),
                    "grad_max_ft": round(grad_max_ft, 2),
                    "extent_ft": (round(rows_extent, 0), round(cols_extent, 0)),
                    "aspect_ratio": round(aspect, 1),
                    "signature": signature,
                    "bag_file": bag_path.name,
                    "pixel_size_ft": round(pixel_size_ft, 2)
                }
                
        except Exception as e:
            return None
    
    def check_known_wreck_match(self, lat, lon):
        """Check if coordinates match a known wreck"""
        for name, info in self.known_wrecks.items():
            dist_nm = ((lat - info['lat'])**2 + (lon - info['lon'])**2)**0.5 * 60
            if dist_nm < 0.1:  # Within 0.1 nm (608 ft)
                return name, dist_nm * NM_TO_FT, info['status']
        return None, None, None
    
    def run_scan(self):
        """Run comprehensive scan"""
        print("=" * 80)
        print("COMPREHENSIVE PDF-BAG CROSS-REFERENCE SCAN")
        print("All measurements in FEET and NAUTICAL MILES")
        print("=" * 80)
        
        bag_files = self.find_bag_files()
        print(f"\nFound {len(bag_files)} BAG files")
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        # Extract all coordinates from all PDFs
        all_coords = []
        print("\n" + "-" * 80)
        print("PHASE 1: EXTRACTING COORDINATES FROM PDFs")
        print("-" * 80)
        
        for pdf_path in sorted(pdf_files):
            coords = self.extract_coordinates_from_pdf(pdf_path)
            print(f"  {pdf_path.name}: {len(coords)} coordinates")
            all_coords.extend(coords)
            self.results["pdfs_processed"].append({
                "name": pdf_path.name,
                "coordinates_found": len(coords)
            })
        
        print(f"\nTotal unique coordinates: {len(all_coords)}")
        self.results["coordinates_extracted"] = all_coords
        
        # Analyze each coordinate against BAG files
        print("\n" + "-" * 80)
        print("PHASE 2: ANALYZING COORDINATES AGAINST BAG FILES")
        print("-" * 80)
        
        analyzed = 0
        no_coverage = 0
        
        for coord in all_coords:
            lat, lon = coord['lat'], coord['lon']
            
            # Check if matches known wreck
            wreck_name, dist_ft, wreck_status = self.check_known_wreck_match(lat, lon)
            
            # Find BAG file
            bag_path = self.find_bag_for_coordinates(lat, lon, bag_files)
            
            if bag_path:
                result = self.analyze_location(bag_path, lat, lon)
                if result:
                    result['lat'] = lat
                    result['lon'] = lon
                    result['pdf'] = coord['pdf']
                    result['page'] = coord['page']
                    
                    if wreck_name:
                        result['known_wreck'] = wreck_name
                        result['dist_to_known_ft'] = round(dist_ft, 0)
                        result['known_status'] = wreck_status
                        self.results['known_wreck_matches'].append(result)
                    
                    if result['signature'] == "WRECK_SIGNATURE":
                        self.results['wreck_signatures'].append(result)
                    elif result['signature'] == "SCRUBBED":
                        self.results['scrubbed_locations'].append(result)
                    elif result['signature'] == "HIGH_TEXTURE":
                        self.results['high_texture'].append(result)
                    
                    analyzed += 1
            else:
                no_coverage += 1
        
        print(f"  Analyzed: {analyzed}")
        print(f"  No BAG coverage: {no_coverage}")
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\n🎯 WRECK SIGNATURES DETECTED: {len(self.results['wreck_signatures'])}")
        for r in self.results['wreck_signatures']:
            wreck_note = f" [{r.get('known_wreck', 'UNKNOWN')}]" if r.get('known_wreck') else " [NEW DISCOVERY?]"
            print(f"  {r['lat']:.5f}, {r['lon']:.5f}{wreck_note}")
            print(f"    Depth: {r['depth_ft']} ft | Extent: {r['extent_ft'][0]:.0f} x {r['extent_ft'][1]:.0f} ft")
            print(f"    Source: {r['pdf']} (page {r['page']})")
        
        print(f"\n🔴 SCRUBBED LOCATIONS (possible hidden wrecks): {len(self.results['scrubbed_locations'])}")
        for r in self.results['scrubbed_locations']:
            wreck_note = f" [HIDDEN {r.get('known_wreck', 'UNKNOWN')}]" if r.get('known_wreck') else " [HIDDEN WRECK?]"
            print(f"  {r['lat']:.5f}, {r['lon']:.5f}{wreck_note}")
            print(f"    Depth: {r['depth_ft']} ft | Std: {r['std_ft']} ft (suspiciously low)")
        
        print(f"\n📊 HIGH TEXTURE AREAS: {len(self.results['high_texture'])}")
        
        print(f"\n✅ KNOWN WRECK MATCHES: {len(self.results['known_wreck_matches'])}")
        for r in self.results['known_wreck_matches']:
            status = "VISIBLE" if r['signature'] in ['WRECK_SIGNATURE', 'HIGH_TEXTURE'] else "SCRUBBED"
            print(f"  {r['known_wreck']}: {status}")
            print(f"    Position: {r['lat']:.5f}, {r['lon']:.5f}")
            print(f"    Depth: {r['depth_ft']} ft | Signature: {r['signature']}")
        
        # Save results
        output_file = f"comprehensive_pdf_bag_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📁 Full results saved to: {output_file}")
        
        return self.results


if __name__ == "__main__":
    scanner = ComprehensivePDFBAGScanner()
    results = scanner.run_scan()
