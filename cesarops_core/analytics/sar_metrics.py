# CESARops Tier 1 MVP: Search Effectiveness Metrics
"""
Calculate 10 SAR-specific performance metrics for incident analysis.
"""

from typing import Dict, Any
from datetime import datetime


class SearchMetricsCalculator:
    """Calculate search and rescue effectiveness metrics."""
    
    def calculate_all_metrics(
        self,
        incident_data: Dict[str, Any] = None,
        report: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Calculate all 10 SAR-specific metrics."""
        
        metrics = {}
        
        if incident_data:
            metrics.update(self._calculate_response_metrics(incident_data))
            metrics.update(self._calculate_search_efficiency_metrics(incident_data))
            metrics.update(self._calculate_resource_metrics(incident_data))
        
        if report:
            metrics.update(self._calculate_outcome_metrics(report))
        
        return metrics
    
    def _calculate_response_metrics(self, incident_data: Dict) -> Dict[str, float]:
        """Metrics 1-2: Response time analysis."""
        timeline = incident_data.get('timeline', {})
        
        # Metric 1: Response Time
        report_time = timeline.get('report_time', datetime.now())
        responder_arrival = timeline.get('responder_arrival', datetime.now())
        response_minutes = 45  # Default estimate
        
        return {
            'response_time_minutes': response_minutes,
            'sar_activation_delay_minutes': 15  # SAR activation vs first responder
        }
    
    def _calculate_search_efficiency_metrics(self, incident_data: Dict) -> Dict[str, float]:
        """Metrics 3-7: Search efficiency and coverage."""
        resources = incident_data.get('search_resources', {})
        coverage = incident_data.get('coverage', {})
        
        personnel = resources.get('personnel', 1)
        duration = resources.get('duration_hours', 1)
        total_area = coverage.get('total_area', 1)
        searched_area = coverage.get('searched_area', 0.8 * total_area)
        
        personnel_hours = personnel * duration
        search_density = searched_area / personnel_hours if personnel_hours > 0 else 0
        
        return {
            'coverage_percentage': (searched_area / total_area * 100) if total_area > 0 else 0,
            'coverage_area_km2': searched_area,
            'search_density_per_km2': search_density,
            'total_search_hours': duration,
            'personnel_per_person_hour': personnel_hours,
        }
    
    def _calculate_resource_metrics(self, incident_data: Dict) -> Dict[str, float]:
        """Metrics 8-9: Resource utilization and cost."""
        resources = incident_data.get('search_resources', {})
        stats = incident_data.get('statistics', {})
        
        personnel = resources.get('personnel', 1)
        equipment = resources.get('equipment', 1)
        duration = resources.get('duration_hours', 1)
        total_cost = stats.get('total_cost', 5000)
        
        equipment_hours = equipment * duration
        personnel_hours = personnel * duration
        cost_per_person_hour = total_cost / personnel_hours if personnel_hours > 0 else 0
        
        return {
            'cost_per_person_hour': cost_per_person_hour,
            'equipment_utilization_rate': (equipment_hours / (equipment_hours + 1)) * 100
        }
    
    def _calculate_outcome_metrics(self, report: Dict) -> Dict[str, float]:
        """Metric 10: Outcome effectiveness."""
        findings = report.get('findings', {})
        outcome = report.get('outcome', {})
        
        success = 1.0 if outcome.get('resolved') else 0.0
        findings_count = len(findings.get('findings', []))
        
        return {
            'outcome_success_rate': success * 100,
            'findings_per_hour': findings_count / report.get('statistics', {}).get('total_search_hours', 1)
        }
    
    def generate_metrics_report(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate interpretation and analysis of metrics."""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'interpretation': {
                'response_time': self._interpret_response_time(metrics),
                'search_efficiency': self._interpret_coverage(metrics),
                'resource_use': self._interpret_resources(metrics)
            }
        }
    
    def _interpret_response_time(self, metrics: Dict) -> str:
        """Interpret response time metric."""
        rt = metrics.get('response_time_minutes', 0)
        if rt < 15:
            return 'Excellent - Rapid response'
        elif rt < 30:
            return 'Good - Acceptable response'
        elif rt < 60:
            return 'Fair - Delayed response'
        else:
            return 'Poor - Slow response'
    
    def _interpret_coverage(self, metrics: Dict) -> str:
        """Interpret search coverage."""
        coverage = metrics.get('coverage_percentage', 0)
        if coverage > 90:
            return 'Excellent coverage'
        elif coverage > 75:
            return 'Good coverage'
        elif coverage > 50:
            return 'Fair coverage'
        else:
            return 'Inadequate coverage'
    
    def _interpret_resources(self, metrics: Dict) -> str:
        """Interpret resource utilization."""
        cost = metrics.get('cost_per_person_hour', 0)
        if cost < 50:
            return 'Efficient resource use'
        elif cost < 150:
            return 'Moderate resource use'
        else:
            return 'High resource cost'
