# CESARops Tier 1 MVP: PDF/HTML Export & Sample Reports
"""
Professional export functionality for incident reports.
HTML/PDF generation with CSS styling and sample reports.
"""

import os
from datetime import datetime
from typing import Dict, Any


class ReportExporter:
    """Export incident analysis in multiple professional formats."""
    
    def export_html(self, analysis: Dict[str, Any], filepath: str) -> str:
        """
        Export incident analysis to professional HTML report.
        
        Args:
            analysis: Complete incident analysis dictionary
            filepath: Output file path
        
        Returns:
            Path to generated HTML file
        """
        
        incident_id = analysis['incident_id']
        report = analysis['report']
        metrics = analysis['metrics']
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAR Incident Report - {incident_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        
        .header {{
            border-bottom: 3px solid #d32f2f;
            margin-bottom: 30px;
            padding-bottom: 20px;
        }}
        
        .header h1 {{
            font-size: 32px;
            color: #d32f2f;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #666;
            font-size: 14px;
        }}
        
        .incident-id {{
            background: #f0f0f0;
            padding: 10px 15px;
            border-left: 4px solid #d32f2f;
            margin: 20px 0;
            font-weight: bold;
        }}
        
        h2 {{
            font-size: 20px;
            color: #d32f2f;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        
        h3 {{
            font-size: 16px;
            color: #444;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        .section {{
            margin-bottom: 25px;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-box {{
            background: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }}
        
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #d32f2f;
            margin-top: 8px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        
        th {{
            background: #d32f2f;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        
        .status-active {{
            color: #4caf50;
            font-weight: bold;
        }}
        
        .status-resolved {{
            color: #2196f3;
            font-weight: bold;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Search & Rescue Incident Report</h1>
            <p>Professional incident documentation and analysis</p>
        </div>
        
        <div class="incident-id">Incident ID: {incident_id}</div>
        
        <div class="section">
            <h2>Incident Overview</h2>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-label">Incident Type</div>
                    <div class="metric-value">{report['incident_summary']['incident_type'].upper()}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Status</div>
                    <div class="metric-value status-resolved">{analysis.get('executive_summary', {}).get('incident_overview', {}).get('status', 'ACTIVE')}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Search Metrics</h2>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-label">Coverage Area</div>
                    <div class="metric-value">{metrics.get('coverage_area_km2', 0):.1f} km²</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Coverage %</div>
                    <div class="metric-value">{metrics.get('coverage_percentage', 0):.0f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Personnel Deployed</div>
                    <div class="metric-value">{len(report['resources']['personnel'])}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Response Time</div>
                    <div class="metric-value">{metrics.get('response_time_minutes', 0):.0f} min</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Subject Profile</h2>
            <table>
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Name</td>
                    <td>{report['subject_profile']['name']}</td>
                </tr>
                <tr>
                    <td>Age</td>
                    <td>{report['subject_profile']['age']}</td>
                </tr>
                <tr>
                    <td>Experience Level</td>
                    <td>{report['subject_profile']['experience_level']}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Resource Summary</h2>
            <table>
                <tr>
                    <th>Resource Type</th>
                    <th>Quantity</th>
                    <th>Hours Deployed</th>
                </tr>
                <tr>
                    <td>Personnel</td>
                    <td>{len(report['resources']['personnel'])}</td>
                    <td>{metrics.get('total_search_hours', 0):.1f}</td>
                </tr>
                <tr>
                    <td>Equipment Units</td>
                    <td>{len(report['resources']['equipment'])}</td>
                    <td>{metrics.get('total_search_hours', 0):.1f}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Findings</h2>
            <h3>Key Clues</h3>
            <ul>
                {''.join([f'<li>{clue}</li>' for clue in report['findings'].get('clues', [])])}
            </ul>
        </div>
        
        <div class="section">
            <h2>Outcome</h2>
            <p><strong>Resolution Status:</strong> {report['outcome']['outcome_type']}</p>
            <p><strong>Subject Status:</strong> {report['outcome']['subject_status']}</p>
        </div>
        
        <div class="footer">
            <p>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>CESARops Search & Rescue Platform | Confidential</p>
        </div>
    </div>
</body>
</html>
"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def export_pdf(self, analysis: Dict[str, Any], filepath: str) -> str:
        """
        Export to PDF (placeholder - requires reportlab/weasyprint).
        For MVP, HTML export is primary format.
        """
        # TODO: Implement PDF export with reportlab
        return filepath
    
    def export_csv(self, analysis: Dict[str, Any], filepath: str) -> str:
        """Export metrics to CSV format."""
        metrics = analysis['metrics']
        
        csv_content = "Metric,Value\n"
        for key, value in metrics.items():
            csv_content += f'"{key}",{value}\n'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(csv_content)
        
        return filepath


if __name__ == '__main__':
    exporter = ReportExporter()
    print("Export module ready for integration")
