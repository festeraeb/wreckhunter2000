#!/usr/bin/env python3
"""
Rosa Fender Case Demonstration with Multi-Modal FCNN System
===========================================================

Demonstrates the complete multi-modal SAR system using the validated Rosa fender case:
- Multi-modal object description with Sentence Transformers
- FCNN dynamic correction with GPS feedback
- Attention-based trajectory prediction
- Comparison with validated hand calculations

Based on:
- Original Rosa case: 42.995°N, 87.845°W at 8 PM Aug 22
- Validated parameters: A=0.06, Stokes=0.0045
- Recovery at South Haven: 42.403°N, 86.275°W  
- Achieved accuracy: 24.3 nm (excellent validation)

Author: GitHub Copilot  
Date: January 7, 2025
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from multimodal_fcnn_system import MultiModalSAROPS, DriftObject
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    print(f"Multi-modal system not available: {e}")
    print("Run comprehensive_installer.py first to install dependencies")
    MULTIMODAL_AVAILABLE = False

class RosaMultiModalDemo:
    """Demonstration of Rosa case with multi-modal enhancements"""
    
    def __init__(self):
        """Initialize Rosa case demonstration"""
        # Rosa case parameters (validated)
        self.rosa_start = (42.995, -87.845)  # Milwaukee area
        self.rosa_end = (42.403, -86.275)    # South Haven
        self.start_time = datetime(2024, 8, 22, 20, 0, 0)  # 8 PM Aug 22
        
        # Validated parameters
        self.validated_windage = 0.06
        self.validated_stokes = 0.0045
        self.achieved_accuracy_nm = 24.3
        
        # Initialize multi-modal system if available
        if MULTIMODAL_AVAILABLE:
            self.sarops = MultiModalSAROPS()
            self.rosa_object_id = None
        else:
            self.sarops = None
            
        print("Rosa Multi-Modal Demonstration Initialized")
        print(f"Start: {self.rosa_start}")
        print(f"End: {self.rosa_end}")
        print(f"Validated accuracy: {self.achieved_accuracy_nm} nm")
    
    def create_rosa_object(self) -> int:
        """Create enhanced Rosa object with detailed description"""
        if not MULTIMODAL_AVAILABLE:
            print("Multi-modal system not available")
            return -1
        
        # Create detailed Rosa fender object
        rosa_object = DriftObject(
            name="Rosa_Fender_Enhanced",
            mass=2.5,  # kg, estimated mass for inflated fender
            area_above_water=0.06,  # m², validated windage parameter
            area_below_water=0.015,  # m², estimated submerged area
            drag_coefficient_air=1.2,  # Typical for cylindrical bluff body
            drag_coefficient_water=0.8,  # Reduced underwater
            lift_coefficient_air=0.3,  # Moderate aerodynamic lift
            lift_coefficient_water=0.1,  # Minimal hydrodynamic lift
            description="""Pink inflatable cylindrical fender with rounded ends, 
                         measuring approximately 30cm diameter by 90cm length. 
                         Constructed of heavy-duty marine-grade PVC with reinforced 
                         end caps. Partially inflated state providing moderate buoyancy 
                         and significant wind exposure. Bright pink color for high 
                         visibility. Smooth surface texture with minimal surface 
                         roughness. Used as boat bumper with good floating characteristics 
                         and moderate windage when deployed.""",
            object_type="marine_fender"
        )
        
        # Add to database
        self.rosa_object_id = self.sarops.add_drift_object(rosa_object)
        print(f"Created enhanced Rosa object with ID: {self.rosa_object_id}")
        
        return self.rosa_object_id
    
    def simulate_environmental_conditions(self, duration_hours: int = 24) -> np.ndarray:
        """
        Simulate realistic environmental conditions for Rosa case
        Based on typical Lake Michigan summer conditions
        """
        timesteps = duration_hours * 6  # 10-minute intervals
        
        # Base environmental conditions (simplified but realistic)
        environmental_data = np.zeros((timesteps, 10))
        
        for i in range(timesteps):
            time_hours = i / 6.0  # Convert to hours
            
            # Wind conditions (moderate westerly winds typical for August)
            wind_speed = 8.0 + 3.0 * np.sin(time_hours * np.pi / 12)  # Diurnal variation
            wind_dir = 270 + 20 * np.sin(time_hours * np.pi / 6)  # Westerly with variation
            
            # Convert to components
            wind_u = wind_speed * np.cos(np.radians(wind_dir))
            wind_v = wind_speed * np.sin(np.radians(wind_dir))
            
            # Current conditions (typical Lake Michigan circulation)
            current_speed = 0.15 + 0.1 * np.sin(time_hours * np.pi / 8)
            current_dir = 180 + 30 * np.sin(time_hours * np.pi / 10)  # Southward with variation
            
            current_u = current_speed * np.cos(np.radians(current_dir))
            current_v = current_speed * np.sin(np.radians(current_dir))
            
            # Store environmental data
            environmental_data[i, 0] = wind_u
            environmental_data[i, 1] = wind_v
            environmental_data[i, 2] = current_u
            environmental_data[i, 3] = current_v
            environmental_data[i, 4] = wind_speed
            environmental_data[i, 5] = wind_dir
            environmental_data[i, 6] = current_speed
            environmental_data[i, 7] = current_dir
            environmental_data[i, 8] = 22.0 + 2.0 * np.sin(time_hours * np.pi / 12)  # Temperature
            environmental_data[i, 9] = time_hours
        
        return environmental_data
    
    def run_multimodal_prediction(self) -> np.ndarray:
        """Run multi-modal trajectory prediction for Rosa case"""
        if not MULTIMODAL_AVAILABLE or self.rosa_object_id is None:
            print("Multi-modal prediction not available")
            return np.array([])
        
        # Generate environmental conditions
        env_data = self.simulate_environmental_conditions(24)
        
        # Run prediction
        print("Running multi-modal trajectory prediction...")
        trajectory = self.sarops.predict_trajectory(
            object_id=self.rosa_object_id,
            initial_position=self.rosa_start,
            environmental_sequence=env_data,
            prediction_horizon=144  # 24 hours in 10-minute steps
        )
        
        print(f"Predicted trajectory: {len(trajectory)} points")
        print(f"Start position: {trajectory[0]}")
        print(f"End position: {trajectory[-1]}")
        
        return trajectory
    
    def simulate_gps_feedback(self, true_trajectory: np.ndarray) -> dict:
        """Simulate GPS observations with realistic noise for FCNN correction"""
        if not MULTIMODAL_AVAILABLE:
            return {}
        
        # Simulate GPS observations every hour (6 timesteps)
        gps_observations = {}
        
        for i in range(0, len(true_trajectory), 6):
            if i >= len(true_trajectory):
                break
            
            # Add realistic GPS noise (±10m typical accuracy)
            gps_noise = np.random.normal(0, 0.0001, 2)  # ~10m in degrees
            observed_pos = true_trajectory[i] + gps_noise
            
            gps_observations[i] = {
                'time_step': i,
                'observed_position': observed_pos,
                'true_position': true_trajectory[i],
                'noise_level': np.linalg.norm(gps_noise)
            }
        
        return gps_observations
    
    def apply_fcnn_corrections(self, predicted_trajectory: np.ndarray, 
                              gps_observations: dict) -> np.ndarray:
        """Apply FCNN dynamic corrections based on GPS feedback"""
        if not MULTIMODAL_AVAILABLE:
            return predicted_trajectory
        
        corrected_trajectory = predicted_trajectory.copy()
        env_data = self.simulate_environmental_conditions(24)
        
        print("Applying FCNN dynamic corrections...")
        
        for time_step, gps_data in gps_observations.items():
            if time_step >= len(predicted_trajectory):
                continue
            
            # Apply correction
            corrected_pos = self.sarops.apply_dynamic_correction(
                object_id=self.rosa_object_id,
                predicted_position=predicted_trajectory[time_step],
                observed_position=gps_data['observed_position'],
                environmental_features=env_data[time_step]
            )
            
            corrected_trajectory[time_step] = corrected_pos
            
            # Apply correction to subsequent predictions (propagate forward)
            if time_step < len(corrected_trajectory) - 1:
                correction_vector = corrected_pos - predicted_trajectory[time_step]
                
                # Apply diminishing correction to future points
                for j in range(time_step + 1, min(time_step + 12, len(corrected_trajectory))):
                    decay_factor = np.exp(-(j - time_step) / 6.0)  # 1-hour decay
                    corrected_trajectory[j] += correction_vector * decay_factor
        
        return corrected_trajectory
    
    def calculate_accuracy(self, predicted_endpoint: tuple) -> float:
        """Calculate accuracy compared to actual recovery location"""
        # Calculate distance using Haversine formula (simplified)
        lat1, lon1 = predicted_endpoint
        lat2, lon2 = self.rosa_end
        
        # Convert to radians
        lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
        lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in nautical miles
        r_nm = 3440.065
        distance_nm = r_nm * c
        
        return distance_nm
    
    def create_visualization(self, results: dict):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Trajectory comparison
        ax1 = axes[0, 0]
        if 'multimodal_trajectory' in results:
            traj = results['multimodal_trajectory']
            ax1.plot(traj[:, 1], traj[:, 0], 'b-', label='Multi-Modal Prediction', linewidth=2)
        
        if 'corrected_trajectory' in results:
            corr_traj = results['corrected_trajectory']
            ax1.plot(corr_traj[:, 1], corr_traj[:, 0], 'r--', label='FCNN Corrected', linewidth=2)
        
        # Add start and end points
        ax1.plot(self.rosa_start[1], self.rosa_start[0], 'go', markersize=10, label='Start (Milwaukee)')
        ax1.plot(self.rosa_end[1], self.rosa_end[0], 'rs', markersize=10, label='Actual Recovery (South Haven)')
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Rosa Fender Trajectory Prediction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GPS corrections
        ax2 = axes[0, 1]
        if 'gps_observations' in results:
            corrections = []
            times = []
            for time_step, gps_data in results['gps_observations'].items():
                corrections.append(gps_data['noise_level'] * 111000)  # Convert to meters
                times.append(time_step / 6.0)  # Convert to hours
            
            ax2.plot(times, corrections, 'ro-', label='GPS Correction Magnitude')
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Correction Distance (m)')
            ax2.set_title('FCNN Dynamic Corrections')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy comparison
        ax3 = axes[1, 0]
        methods = ['Validated\nHand Calc', 'Multi-Modal\nPrediction']
        accuracies = [self.achieved_accuracy_nm]
        
        if 'multimodal_accuracy' in results:
            accuracies.append(results['multimodal_accuracy'])
        
        if 'corrected_accuracy' in results:
            methods.append('FCNN\nCorrected')
            accuracies.append(results['corrected_accuracy'])
        
        bars = ax3.bar(methods, accuracies, color=['green', 'blue', 'red'])
        ax3.set_ylabel('Accuracy (nautical miles)')
        ax3.set_title('Prediction Accuracy Comparison')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f} nm', ha='center', va='bottom')
        
        # Plot 4: Performance summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""
Rosa Fender Case Summary
========================

Original Case:
• Start: {self.rosa_start[0]:.3f}°N, {self.rosa_start[1]:.3f}°W
• End: {self.rosa_end[0]:.3f}°N, {self.rosa_end[1]:.3f}°W
• Date: August 22, 2024 8:00 PM

Validated Parameters:
• Windage (A): {self.validated_windage}
• Stokes Drift: {self.validated_stokes}
• Hand Calculation Accuracy: {self.achieved_accuracy_nm} nm

Multi-Modal Enhancements:
• Sentence Transformer object description
• FCNN dynamic correction with GPS feedback
• Attention-based trajectory prediction
• Variable immersion ratio modeling
"""
        
        if 'multimodal_accuracy' in results:
            summary_text += f"\nResults:\n• Multi-Modal Accuracy: {results['multimodal_accuracy']:.1f} nm"
        
        if 'corrected_accuracy' in results:
            summary_text += f"\n• FCNN Corrected Accuracy: {results['corrected_accuracy']:.1f} nm"
            improvement = self.achieved_accuracy_nm - results['corrected_accuracy']
            summary_text += f"\n• Improvement: {improvement:.1f} nm ({improvement/self.achieved_accuracy_nm*100:.1f}%)"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('rosa_multimodal_demo_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_demonstration(self):
        """Run complete Rosa case demonstration"""
        print("\n" + "="*60)
        print("ROSA FENDER MULTI-MODAL DEMONSTRATION")
        print("="*60)
        
        results = {}
        
        # Step 1: Create enhanced Rosa object
        if MULTIMODAL_AVAILABLE:
            self.create_rosa_object()
            
            # Step 2: Run multi-modal prediction
            predicted_trajectory = self.run_multimodal_prediction()
            if len(predicted_trajectory) > 0:
                results['multimodal_trajectory'] = predicted_trajectory
                
                # Calculate accuracy
                multimodal_accuracy = self.calculate_accuracy(predicted_trajectory[-1])
                results['multimodal_accuracy'] = multimodal_accuracy
                print(f"Multi-modal prediction accuracy: {multimodal_accuracy:.1f} nm")
                
                # Step 3: Simulate GPS feedback
                gps_observations = self.simulate_gps_feedback(predicted_trajectory)
                results['gps_observations'] = gps_observations
                print(f"Simulated {len(gps_observations)} GPS observations")
                
                # Step 4: Apply FCNN corrections
                corrected_trajectory = self.apply_fcnn_corrections(
                    predicted_trajectory, gps_observations
                )
                results['corrected_trajectory'] = corrected_trajectory
                
                # Calculate corrected accuracy
                corrected_accuracy = self.calculate_accuracy(corrected_trajectory[-1])
                results['corrected_accuracy'] = corrected_accuracy
                print(f"FCNN corrected accuracy: {corrected_accuracy:.1f} nm")
                
                # Calculate improvement
                improvement = self.achieved_accuracy_nm - corrected_accuracy
                print(f"Improvement over hand calculation: {improvement:.1f} nm ({improvement/self.achieved_accuracy_nm*100:.1f}%)")
        
        else:
            print("Multi-modal system not available - install dependencies first")
            print("Run: python comprehensive_installer.py")
        
        # Step 5: Create visualization
        self.create_visualization(results)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("Results saved to: rosa_multimodal_demo_results.png")
        
        return results

def main():
    """Main demonstration function"""
    print("Rosa Fender Multi-Modal Demonstration")
    print("=====================================")
    
    # Check if dependencies are available
    if not MULTIMODAL_AVAILABLE:
        print("\nDependencies not installed!")
        print("Please run the installer first:")
        print("  python comprehensive_installer.py")
        print("\nThen activate the environment:")
        print("  conda activate cesarops")
        print("  python rosa_multimodal_demo.py")
        return
    
    # Run demonstration
    demo = RosaMultiModalDemo()
    results = demo.run_complete_demonstration()
    
    # Optional: Save results to file
    if results:
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = value
            else:
                json_results[key] = value
        
        with open('rosa_demo_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print("Detailed results saved to: rosa_demo_results.json")

if __name__ == "__main__":
    main()