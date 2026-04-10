# CESARops Tier 1 MVP: Incident Report Generator
"""
Professional incident documentation module for SAR operations.
Generates 10-section comprehensive reports with JSON/HTML export.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional


class IncidentReport:
    """Generate comprehensive incident documentation for SAR operations."""
    
    def __init__(self, db_file: str = 'reports.db'):
        """Initialize incident report generator."""
        self.db_file = db_file
    
    def generate_report(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive 10-section incident report.
        
        Sections:
        1. Header & Metadata
        2. Incident Summary
        3. Subject Profile
        4. Search Timeline
        5. Resources Deployed
        6. Search Coverage
        7. Findings & Clues
        8. Communications Log
        9. Outcome & Resolution
        10. Statistics & Analysis
        """
        
        report = {
            'header': self._generate_header(incident_data),
            'incident_summary': self._generate_incident_summary(incident_data),
            'subject_profile': self._generate_subject_profile(incident_data),
            'timeline': self._generate_timeline(incident_data),
            'resources': self._generate_resources_section(incident_data),
            'search_coverage': self._generate_coverage_section(incident_data),
            'findings': self._generate_findings_section(incident_data),
            'communications_log': self._generate_communications_log(incident_data),
            'outcome': self._generate_outcome_section(incident_data),
            'statistics': self._generate_statistics_section(incident_data)
        }
        
        return report
    
    def _generate_header(self, incident_data: Dict) -> Dict[str, Any]:
        """Section 1: Report header and metadata."""
        return {
            'incident_id': incident_data.get('incident_id', 'UNKNOWN'),
            'report_date': datetime.now().isoformat(),
            'reporting_authority': incident_data.get('authority', 'SAR Operations Center'),
            'jurisdiction': incident_data.get('jurisdiction', 'Unknown'),
            'classification': incident_data.get('classification', 'Search & Rescue')
        }
    
    def _generate_incident_summary(self, incident_data: Dict) -> Dict[str, Any]:
        """Section 2: Incident overview and summary."""
        return {
            'incident_type': incident_data.get('incident_type', 'unknown'),
            'location': incident_data.get('location', (0, 0)),
            'timestamp': incident_data.get('timestamp', datetime.now().isoformat()),
            'weather_conditions': incident_data.get('weather', 'Fair'),
            'terrain': incident_data.get('terrain', 'Mixed'),
            'urgency_level': incident_data.get('urgency', 'High'),
            'search_area_bounds': incident_data.get('search_area_bounds', [])
        }
    
    def _generate_subject_profile(self, incident_data: Dict) -> Dict[str, Any]:
        """Section 3: Missing subject profile and characteristics."""
        subject = incident_data.get('subject_profile', {})
        return {
            'name': subject.get('name', 'Subject'),
            'age': subject.get('age', 'Unknown'),
            'gender': subject.get('gender', 'Unknown'),
            'physical_description': subject.get('description', 'Not provided'),
            'medical_conditions': subject.get('medical_conditions', []),
            'experience_level': subject.get('experience', 'Unknown'),
            'equipment_carrying': subject.get('equipment', []),
            'last_known_condition': subject.get('condition', 'Unknown')
        }
    
    def _generate_timeline(self, incident_data: Dict) -> Dict[str, Any]:
        """Section 4: Chronological event timeline."""
        timeline = incident_data.get('timeline', {})
        return {
            'initial_report_time': timeline.get('report_time', datetime.now().isoformat()),
            'first_responder_arrival': timeline.get('responder_arrival', None),
            'sar_activation_time': timeline.get('sar_activation', None),
            'initial_search_start': timeline.get('search_start', None),
            'last_known_position_time': timeline.get('last_position_time', None),
            'significant_events': timeline.get('events', [])
        }
    
    def _generate_resources_section(self, incident_data: Dict) -> Dict[str, Any]:
        """Section 5: Resources deployed in search operation."""
        resources = incident_data.get('search_resources', {})
        return {
            'personnel': [
                {
                    'name': p.get('name', f'Personnel {i}'),
                    'role': p.get('role', 'Search Team Member'),
                    'hours_worked': p.get('hours', 0),
                    'specialized_training': p.get('training', []),
                    'lat': p.get('lat'),
                    'lon': p.get('lon')
                }
                for i, p in enumerate(resources.get('personnel_list', []), 1)
            ],
            'equipment': [
                {
                    'type': e.get('type', 'Generic Equipment'),
                    'quantity': e.get('quantity', 1),
                    'status': e.get('status', 'Active'),
                    'deployment_location': e.get('location', 'Unknown')
                }
                for e in resources.get('equipment_list', [])
            ],
            'total_personnel': resources.get('personnel', 0),
            'total_equipment_units': resources.get('equipment', 0),
            'search_duration_hours': resources.get('duration_hours', 0)
        }
    
    def _generate_coverage_section(self, incident_data: Dict) -> Dict[str, Any]:
        """Section 6: Search area coverage analysis."""
        coverage = incident_data.get('coverage', {})
        return {
            'total_area_km2': coverage.get('total_area', 0),
            'searched_area_km2': coverage.get('searched_area', 0),
            'coverage_percentage': coverage.get('coverage_pct', 0),
            'search_density_per_km2': coverage.get('search_density', 0),
            'areas_not_searched': coverage.get('not_searched', []),
            'search_pattern': coverage.get('pattern', 'Systematic grid'),
            'terrain_challenges': coverage.get('challenges', [])
        }
    
    def _generate_findings_section(self, incident_data: Dict) -> Dict[str, Any]:
        """Section 7: Findings, clues, and evidence."""
        findings = incident_data.get('findings', {})
        return {
            'clues': findings.get('clues', []),
            'findings': findings.get('items', []),
            'false_alarms': findings.get('false_alarms', 0),
            'evidence_collected': findings.get('evidence', []),
            'notable_observations': findings.get('observations', []),
            'search_effectiveness': findings.get('effectiveness', 'Fair')
        }
    
    def _generate_communications_log(self, incident_data: Dict) -> List[Dict[str, Any]]:
        """Section 8: Communication events log."""
        comms = incident_data.get('communications', [])
        return [
            {
                'timestamp': c.get('timestamp', datetime.now().isoformat()),
                'from_unit': c.get('from', 'Unknown'),
                'to_unit': c.get('to', 'All Units'),
                'message': c.get('message', ''),
                'message_type': c.get('type', 'Status Update'),
                'priority': c.get('priority', 'Normal')
            }
            for c in comms
        ]
    
    def _generate_outcome_section(self, incident_data: Dict) -> Dict[str, Any]:
        """Section 9: Outcome and resolution details."""
        outcome = incident_data.get('outcome', {})
        return {
            'outcome_type': outcome.get('type', 'Suspended'),
            'resolved': outcome.get('resolved', False),
            'subject_status': outcome.get('subject_status', 'Unknown'),
            'resolution_date': outcome.get('resolution_date', None),
            'subject_condition': outcome.get('subject_condition', 'Unknown'),
            'rescue_method': outcome.get('rescue_method', 'N/A'),
            'contributing_factors': outcome.get('factors', [])
        }
    
    def _generate_statistics_section(self, incident_data: Dict) -> Dict[str, Any]:
        """Section 10: Summary statistics and analysis."""
        stats = incident_data.get('statistics', {})
        resources = incident_data.get('search_resources', {})
        
        total_personnel_hours = resources.get('personnel', 0) * resources.get('duration_hours', 0)
        total_cost = stats.get('total_cost', 0)
        
        return {
            'total_search_hours': resources.get('duration_hours', 0),
            'total_personnel_hours': total_personnel_hours,
            'total_equipment_hours': resources.get('equipment', 0) * resources.get('duration_hours', 0),
            'total_cost': total_cost,
            'cost_per_personnel_hour': total_cost / total_personnel_hours if total_personnel_hours > 0 else 0,
            'personnel_efficiency': stats.get('efficiency', 'Fair'),
            'success_metrics': stats.get('success_metrics', {}),
            'lessons_learned': stats.get('lessons', [])
        }
    
    def retrieve_report(self, incident_id: str) -> Dict[str, Any]:
        """Retrieve previously generated report."""
        # This would query database in production
        return {'incident_id': incident_id, 'status': 'retrieved'}


if __name__ == '__main__':
    generator = IncidentReport()
    test_incident = {
        'incident_id': 'TEST-LAND-001',
        'incident_type': 'land',
        'location': (37.7749, -122.4194),
        'subject_profile': {'name': 'John Doe', 'age': 45, 'experience': 'Intermediate'},
        'search_resources': {'personnel': 28, 'equipment': 8, 'duration_hours': 24}
    }
    report = generator.generate_report(test_incident)
    print(f"Report generated: {report['header']['incident_id']}")
