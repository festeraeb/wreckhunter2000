# CESARops Tier 1 MVP: Unified Reporting API
"""
High-level API orchestration for all reporting functionality.
Combines incident reports, metrics, and export capabilities.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from incident_report import IncidentReport
from sar_metrics import SearchMetricsCalculator
from pdf_export import ReportExporter


class CESAROpsReportingAPI:
    """
    Unified API for incident reporting, metrics, and export.
    Single interface for all SAR reporting operations.
    """
    
    def __init__(self, db_file: str = 'reports.db'):
        """Initialize reporting API with database connection."""
        self.db_file = db_file
        self.incident_report = IncidentReport(db_file)
        self.metrics = SearchMetricsCalculator()
        self.exporter = ReportExporter()
    
    def create_incident_analysis(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete incident analysis from raw data.
        
        Args:
            incident_data: Dictionary with incident details
                - incident_id: Unique identifier
                - incident_type: 'land', 'water', 'k9', 'aerial', 'dive', 'equine'
                - location: (lat, lon) tuple
                - subject_profile: Name, age, experience, equipment
                - search_resources: Personnel, equipment, duration
                - findings: Search results, clues, bodies
        
        Returns:
            Complete incident analysis with report, metrics, and export data
        """
        
        # Generate incident report
        report = self.incident_report.generate_report(incident_data)
        
        # Calculate metrics
        metrics = self.metrics.calculate_all_metrics(
            incident_data=incident_data,
            report=report
        )
        
        # Generate executive summary
        summary = self._generate_executive_summary(report, metrics)
        
        return {
            'incident_id': incident_data['incident_id'],
            'timestamp': datetime.now().isoformat(),
            'report': report,
            'metrics': metrics,
            'executive_summary': summary,
            'dashboard_data': self._prepare_dashboard_data(report, metrics)
        }
    
    def export_incident_analysis(
        self,
        analysis: Dict[str, Any],
        formats: List[str] = None,
        output_dir: str = 'reports/generated'
    ) -> Dict[str, str]:
        """
        Export incident analysis in multiple formats.
        
        Args:
            analysis: Complete incident analysis from create_incident_analysis
            formats: List of export formats: ['html', 'json', 'pdf', 'csv']
            output_dir: Directory for output files
        
        Returns:
            Dictionary mapping format -> file path
        """
        if formats is None:
            formats = ['html', 'json']
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        incident_id = analysis['incident_id']
        exported_files = {}
        
        for fmt in formats:
            if fmt == 'html':
                filepath = self.exporter.export_html(
                    analysis,
                    os.path.join(output_dir, f"{incident_id}_report.html")
                )
                exported_files['html'] = filepath
            
            elif fmt == 'json':
                filepath = os.path.join(output_dir, f"{incident_id}_analysis.json")
                with open(filepath, 'w') as f:
                    # Make report JSON serializable
                    json_data = self._make_json_serializable(analysis)
                    json.dump(json_data, f, indent=2)
                exported_files['json'] = filepath
            
            elif fmt == 'pdf':
                filepath = self.exporter.export_pdf(
                    analysis,
                    os.path.join(output_dir, f"{incident_id}_report.pdf")
                )
                exported_files['pdf'] = filepath
            
            elif fmt == 'csv':
                filepath = self.exporter.export_csv(
                    analysis,
                    os.path.join(output_dir, f"{incident_id}_metrics.csv")
                )
                exported_files['csv'] = filepath
        
        return exported_files
    
    def generate_dashboard_data(self, incident_id: str) -> Dict[str, Any]:
        """
        Generate real-time dashboard data for command center display.
        
        Args:
            incident_id: Incident identifier
        
        Returns:
            Dashboard-formatted data ready for visualization
        """
        # Retrieve incident data
        report = self.incident_report.retrieve_report(incident_id)
        metrics = self.metrics.calculate_all_metrics({'incident_id': incident_id})
        
        return self._prepare_dashboard_data(report, metrics)
    
    def _generate_executive_summary(
        self,
        report: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary from report and metrics."""
        
        return {
            'incident_overview': {
                'type': report['incident_summary']['incident_type'],
                'location': report['incident_summary']['location'],
                'timestamp': report['incident_summary']['timestamp'],
                'status': 'Active' if not report['outcome']['resolved'] else 'Resolved'
            },
            'search_summary': {
                'total_personnel': len(report['resources']['personnel']),
                'total_equipment': len(report['resources']['equipment']),
                'area_searched': metrics['coverage_area_km2'],
                'search_duration_hours': metrics['total_search_hours'],
                'coverage_percentage': metrics['coverage_percentage']
            },
            'key_findings': {
                'subject_status': report['outcome']['outcome_type'],
                'notable_clues': report['findings']['clues'][:3] if report['findings']['clues'] else [],
                'response_time_minutes': metrics['response_time_minutes']
            },
            'resource_utilization': {
                'personnel_efficiency': metrics['personnel_per_person_hour'],
                'cost_per_person_hour': metrics['cost_per_person_hour'],
                'equipment_utilization_rate': metrics['equipment_utilization_rate']
            },
            'recommendations': self._generate_recommendations(report, metrics)
        }
    
    def _prepare_dashboard_data(
        self,
        report: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data formatted for dashboard visualization."""
        
        return {
            'incident_id': report['incident_id'],
            'status': 'Active' if not report['outcome']['resolved'] else 'Resolved',
            'subject': {
                'name': report['subject_profile']['name'],
                'age': report['subject_profile']['age'],
                'last_seen': report['timeline']['initial_report_time'],
                'location': report['incident_summary']['location']
            },
            'search_stats': {
                'elapsed_time': metrics['total_search_hours'],
                'personnel_deployed': len(report['resources']['personnel']),
                'equipment_units': len(report['resources']['equipment']),
                'area_searched_km2': metrics['coverage_area_km2'],
                'coverage_percent': min(100, metrics['coverage_percentage'])
            },
            'key_metrics': {
                'response_time': metrics['response_time_minutes'],
                'search_density': metrics['search_density_per_km2'],
                'findings_count': len(report['findings']['findings']),
                'utilization_rate': metrics['equipment_utilization_rate']
            },
            'timeline': [
                {'time': report['timeline']['initial_report_time'], 'event': 'Initial Report'},
                {'time': report['timeline']['first_responder_arrival'], 'event': 'First Responder Arrival'},
                {'time': report['timeline']['sar_activation_time'], 'event': 'SAR Activation'},
            ] + [
                {'time': c['timestamp'], 'event': c['message'][:50]}
                for c in report['communications_log'][-5:]  # Last 5 comms
            ],
            'map_data': {
                'center': report['incident_summary']['location'],
                'search_area': report['incident_summary'].get('search_area_bounds', []),
                'personnel_positions': [
                    {'lat': p.get('lat', 0), 'lon': p.get('lon', 0), 'name': p['name']}
                    for p in report['resources']['personnel'][:10]
                ]
            }
        }
    
    def _generate_recommendations(
        self,
        report: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on incident outcome."""
        recommendations = []
        
        # Response time recommendations
        if metrics['response_time_minutes'] > 60:
            recommendations.append(
                'Response time exceeded 1 hour. Consider pre-staging resources.'
            )
        
        # Coverage recommendations
        if metrics['coverage_percentage'] < 80:
            recommendations.append(
                'Search coverage < 80%. Recommend expanding search area or deploying additional resources.'
            )
        
        # Personnel efficiency
        if metrics['personnel_per_person_hour'] > 200:
            recommendations.append(
                'High personnel hours. Consider more efficient search patterns or equipment (drones, etc).'
            )
        
        # Equipment utilization
        if metrics['equipment_utilization_rate'] < 50:
            recommendations.append(
                'Equipment underutilized. Consider reducing equipment or redeploying to other areas.'
            )
        
        return recommendations
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return str(obj)


# Example usage and test
if __name__ == '__main__':
    api = CESAROpsReportingAPI()
    
    # Sample incident data
    test_incident = {
        'incident_id': 'TEST-001',
        'incident_type': 'land',
        'location': (37.7749, -122.4194),
        'search_resources': {
            'personnel': 28,
            'equipment': 8,
            'search_duration_hours': 24
        }
    }
    
    # Create analysis
    analysis = api.create_incident_analysis(test_incident)
    print(f"Analysis created: {analysis['incident_id']}")
    print(f"Coverage: {analysis['metrics']['coverage_percentage']:.1f}%")
    
    # Export to multiple formats
    exports = api.export_incident_analysis(analysis)
    print(f"Exports: {list(exports.keys())}")
