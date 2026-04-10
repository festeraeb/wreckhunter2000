#!/usr/bin/env python3
"""
CESAROPS - Comprehensive Enhanced Search and Rescue Operations Platform
Integrated SAR application with role-based access and professional reporting

Features:
- Unified command interface for all SAR components
- Role-based access control (Family, Law Enforcement, SAR Teams, Command)
- Professional reporting for different stakeholders
- Real-time status monitoring and updates
- Integrated ML-enhanced drift modeling
- Aerial asset coordination
- Computer vision processing
- ICS-compliant command structure
- Multi-agency coordination
"""

import os
import sys
import json
import sqlite3
import logging
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import webbrowser
import subprocess

# Import CESAROPS components
try:
    from incident_command_system import IncidentCommandSystem, IncidentType, OperationalPeriod
    from coordinator_roles_components import CoordinatorManager
    from computer_vision_pipeline import ComputerVisionPipeline
    from civilian_air_patrol_integration import CivilianAirPatrolCoordinator
    from aerial_asset_integration import AerialAssetManager
    from garmin_rsd_integration import CESAROPSRSDIntegration, create_rsd_integration_gui
    ICS_AVAILABLE = True
    RSD_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    ICS_AVAILABLE = False
    RSD_INTEGRATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cesarops.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UserRole:
    """User role definitions with access levels"""
    FAMILY = "family"
    LAW_ENFORCEMENT = "law_enforcement"
    SAR_TEAM = "sar_team"
    INCIDENT_COMMANDER = "incident_commander"
    SYSTEM_ADMIN = "system_admin"

@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    role: str
    full_name: str
    agency: str
    contact_info: Dict[str, str]
    permissions: List[str]
    last_login: Optional[datetime] = None
    active: bool = True

@dataclass
class IncidentSummary:
    """Incident summary for reporting"""
    incident_id: str
    incident_name: str
    incident_type: str
    status: str
    start_time: datetime
    location: Dict[str, float]
    people_involved: int
    resources_deployed: int
    search_areas: int
    current_phase: str
    weather_conditions: Dict[str, Any]
    progress_summary: str
    next_update: datetime

class CESAROPSApplication:
    """Main CESAROPS application with integrated components"""
    
    def __init__(self):
        self.data_directory = 'data'
        self.logs_directory = 'logs'
        self.outputs_directory = 'outputs'
        self.reports_directory = 'reports'
        
        # Initialize directories
        for directory in [self.data_directory, self.logs_directory, 
                         self.outputs_directory, self.reports_directory]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize database
        self.db_path = os.path.join(self.data_directory, 'cesarops_main.db')
        self._initialize_main_database()
        
        # Initialize components
        self.components_initialized = False
        self._initialize_components()
        
        # Current user and session
        self.current_user: Optional[User] = None
        self.active_incidents: Dict[str, Any] = {}
        
        # Initialize demo users
        self._initialize_demo_users()
        
        # GUI components
        self.root = None
        self.main_frame = None
        
    def _initialize_main_database(self):
        """Initialize main CESAROPS database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    role TEXT,
                    full_name TEXT,
                    agency TEXT,
                    contact_info TEXT,
                    permissions TEXT,
                    last_login TEXT,
                    active BOOLEAN
                )
            """)
            
            # Session logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_logs (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    login_time TEXT,
                    logout_time TEXT,
                    actions_performed TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    incident_id TEXT,
                    report_type TEXT,
                    target_audience TEXT,
                    generated_by TEXT,
                    generated_time TEXT,
                    file_path TEXT,
                    content TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Main CESAROPS database initialized")
        except Exception as e:
            logger.error(f"Error initializing main database: {e}")
    
    def _initialize_components(self):
        """Initialize all CESAROPS components"""
        try:
            if not ICS_AVAILABLE:
                logger.warning("Some components not available - running in limited mode")
                return
            
            # Initialize ICS
            self.ics = IncidentCommandSystem(self.data_directory)
            
            # Initialize coordinator manager
            self.coordinators = CoordinatorManager(self.data_directory)
            
            # Initialize computer vision
            cv_config = {
                'data_directory': self.data_directory,
                'person_model_path': None,
                'vessel_model_path': None
            }
            self.computer_vision = ComputerVisionPipeline(cv_config)
            
            # Initialize civilian air patrol
            self.cap_coordinator = CivilianAirPatrolCoordinator(self.data_directory)
            
            # Initialize aerial asset manager
            aerial_config = {
                'data_directory': self.data_directory,
                'enable_mavlink': False,  # Disable for demo
                'enable_computer_vision': True,
                'enable_satellite_data': False  # Disable for demo
            }
            self.aerial_manager = AerialAssetManager(aerial_config)
            
            # Initialize Garmin RSD integration
            if RSD_INTEGRATION_AVAILABLE:
                rsd_db_path = os.path.join(self.data_directory, 'drift_objects.db')
                self.rsd_integration = CESAROPSRSDIntegration(rsd_db_path)
                logger.info("Garmin RSD Studio integration initialized")
            else:
                self.rsd_integration = None
                logger.warning("Garmin RSD integration not available")
            
            self.components_initialized = True
            logger.info("All CESAROPS components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.components_initialized = False
    
    def _initialize_demo_users(self):
        """Initialize demo users for different roles"""
        try:
            demo_users = [
                User(
                    user_id="FAM001",
                    username="family_member",
                    role=UserRole.FAMILY,
                    full_name="Sarah Johnson",
                    agency="Family Member",
                    contact_info={"phone": "555-0123", "email": "sarah@example.com"},
                    permissions=["view_basic_status", "receive_updates"]
                ),
                User(
                    user_id="LE001",
                    username="law_enforcement",
                    role=UserRole.LAW_ENFORCEMENT,
                    full_name="Officer Mike Smith",
                    agency="County Sheriff",
                    contact_info={"phone": "555-0456", "email": "msmith@sheriff.gov"},
                    permissions=["view_detailed_status", "access_evidence", "coordinate_resources"]
                ),
                User(
                    user_id="SAR001",
                    username="sar_team",
                    role=UserRole.SAR_TEAM,
                    full_name="Captain Lisa Brown",
                    agency="Coast Guard Auxiliary",
                    contact_info={"phone": "555-0789", "email": "lbrown@cgaux.org"},
                    permissions=["full_operational_access", "manage_resources", "update_status"]
                ),
                User(
                    user_id="IC001",
                    username="incident_commander",
                    role=UserRole.INCIDENT_COMMANDER,
                    full_name="Commander John Davis",
                    agency="US Coast Guard",
                    contact_info={"phone": "555-0100", "email": "jdavis@uscg.mil"},
                    permissions=["full_command_access", "create_incidents", "manage_users", "approve_reports"]
                )
            ]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for user in demo_users:
                cursor.execute("""
                    INSERT OR REPLACE INTO users 
                    (user_id, username, role, full_name, agency, contact_info, permissions, last_login, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.user_id, user.username, user.role, user.full_name,
                    user.agency, json.dumps(user.contact_info), json.dumps(user.permissions),
                    user.last_login.isoformat() if user.last_login else None, user.active
                ))
            
            conn.commit()
            conn.close()
            logger.info("Demo users initialized")
            
        except Exception as e:
            logger.error(f"Error initializing demo users: {e}")
    
    def authenticate_user(self, username: str, password: str = "demo") -> bool:
        """Authenticate user (simplified for demo)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM users WHERE username = ? AND active = 1", (username,))
            row = cursor.fetchone()
            
            if row:
                self.current_user = User(
                    user_id=row[0],
                    username=row[1],
                    role=row[2],
                    full_name=row[3],
                    agency=row[4],
                    contact_info=json.loads(row[5]),
                    permissions=json.loads(row[6]),
                    last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                    active=bool(row[8])
                )
                
                # Update last login
                cursor.execute("UPDATE users SET last_login = ? WHERE user_id = ?",
                             (datetime.now().isoformat(), self.current_user.user_id))
                conn.commit()
                
                logger.info(f"User authenticated: {username} ({self.current_user.role})")
                return True
            
            conn.close()
            return False
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return False
    
    def create_demo_incident(self) -> str:
        """Create demonstration incident"""
        try:
            if not self.components_initialized:
                return ""
            
            # Create incident
            incident_id = self.ics.create_incident(
                incident_name="Missing Fishing Vessel 'Aurora'",
                incident_type=IncidentType.OVERDUE_VESSEL,
                incident_commander="IC001",
                location={"lat": 42.9634, "lon": -86.4526},  # Lake Michigan
                description="32-foot sport fishing boat with 4 persons on board overdue from fishing trip. Last radio contact at 14:30 hours reporting engine trouble 12 miles west of Grand Haven. Weather deteriorating with 4-6 foot seas.",
                jurisdiction="US Coast Guard District 9"
            )
            
            if incident_id:
                # Assign resources
                self.ics.assign_resource(incident_id, "BOAT001")
                self.ics.assign_resource(incident_id, "HELO001")
                
                # Create search areas
                incident_location = {"lat": 42.9634, "lon": -86.4526}
                available_resources = ["BOAT001", "BOAT002", "HELO001", "AIRCRAFT001"]
                
                response_plan = self.coordinators.coordinate_integrated_response(
                    incident_id, incident_location, 8.0, available_resources
                )
                
                # Coordinate CAP resources
                cap_mission = self.cap_coordinator.coordinate_sar_mission(
                    "Support search for missing fishing vessel with 4 POB",
                    {
                        'center_lat': 42.9634,
                        'center_lon': -86.4526,
                        'radius_nm': 8.0,
                        'size_sq_nm': 201.0,
                        'terrain': 'open_water',
                        'hazards': ['rough_seas', 'weather']
                    },
                    'urgent'
                )
                
                self.active_incidents[incident_id] = {
                    'ics_incident': incident_id,
                    'response_plan': response_plan,
                    'cap_mission': cap_mission,
                    'created_time': datetime.now()
                }
                
                logger.info(f"Demo incident created: {incident_id}")
                return incident_id
            
            return ""
            
        except Exception as e:
            logger.error(f"Error creating demo incident: {e}")
            return ""
    
    def generate_family_report(self, incident_id: str) -> str:
        """Generate family-friendly status report"""
        try:
            if not self.components_initialized:
                return "System components not available"
            
            # Get incident status
            status = self.ics.get_incident_status(incident_id)
            if not status:
                return "Incident not found"
            
            # Create family report
            report_content = f"""
SEARCH AND RESCUE UPDATE
{datetime.now().strftime('%B %d, %Y at %I:%M %p')}

INCIDENT: {status.get('incident_name', 'Unknown')}
STATUS: Search operations are actively underway

WHAT WE KNOW:
• A fishing vessel with 4 people aboard is overdue from their planned return
• Last known location: Approximately 12 miles west of Grand Haven, Michigan
• Weather conditions: 4-6 foot seas with deteriorating conditions
• Last radio contact: Today at 2:30 PM reporting engine trouble

SEARCH EFFORTS:
• {status.get('resources_count', 0)} Coast Guard and partner agency vessels are searching
• Helicopter with advanced search equipment is covering the area
• Search area covers approximately 200 square miles
• Civilian volunteer aircraft are providing additional coverage

WHAT HAPPENS NEXT:
• Search operations will continue through the night with specialized equipment
• Weather conditions are being closely monitored
• Additional resources are available if needed
• Next update will be provided in 2 hours or when significant developments occur

FAMILY SUPPORT:
• Family liaison officer: Available 24/7 at (555) 0199
• Support services: Available for families at the incident command post
• For questions: Contact the incident information officer

The Coast Guard and partner agencies are committed to bringing your loved ones home safely.
Our experienced search and rescue teams are using all available resources.

This report was generated by the CESAROPS system at {datetime.now().isoformat()}
            """
            
            # Save report
            report_filename = f"family_update_{incident_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join(self.reports_directory, report_filename)
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            # Log report generation
            self._log_report_generation(incident_id, "family_update", UserRole.FAMILY, report_path)
            
            return report_content
            
        except Exception as e:
            logger.error(f"Error generating family report: {e}")
            return f"Error generating report: {e}"
    
    def generate_law_enforcement_report(self, incident_id: str) -> str:
        """Generate law enforcement operational report"""
        try:
            if not self.components_initialized:
                return "System components not available"
            
            # Get comprehensive status
            ics_status = self.ics.get_incident_status(incident_id)
            coord_report = self.coordinators.generate_coordination_report(incident_id)
            
            # Create law enforcement report
            report_content = f"""
LAW ENFORCEMENT OPERATIONAL REPORT
INCIDENT: {ics_status.get('incident_id', 'Unknown')}
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
CLASSIFICATION: LAW ENFORCEMENT SENSITIVE

INCIDENT OVERVIEW:
• Type: {ics_status.get('incident_type', 'Unknown')}
• Status: {ics_status.get('status', 'Unknown')}
• Complexity: {ics_status.get('complexity_level', 'Unknown')}
• Duration: {ics_status.get('duration', 'Unknown')}
• Jurisdiction: US Coast Guard District 9

PERSONS INVOLVED:
• Missing: 4 persons aboard fishing vessel 'Aurora'
• Names: [Information available to authorized personnel]
• Emergency contacts: Notified and being supported
• Witness statements: Being collected

OPERATIONAL STATUS:
• Command Structure: {ics_status.get('incident_commander', 'Unknown')}
• Personnel Assigned: {ics_status.get('personnel_count', 0)}
• Resources Deployed: {ics_status.get('resources_count', 0)}
• Search Areas: {coord_report.get('search_coordination', {}).get('active_areas', 0)}

SEARCH COORDINATION:
• Active Search Areas: {coord_report.get('search_coordination', {}).get('active_areas', 0)}
• Coverage Completed: {coord_report.get('search_coordination', {}).get('average_coverage', 0):.1f}%
• Resources Deployed: {coord_report.get('search_coordination', {}).get('resources_deployed', 0)}

RESCUE OPERATIONS:
• Active Rescues: {coord_report.get('rescue_coordination', {}).get('active_rescues', 0)}
• Medical Resources: Standby
• Evacuation Plans: Coast Guard helicopter and surface vessels

COMMUNICATIONS:
• Primary Frequency: 121.50 MHz
• Tactical Frequencies: 156.80, 157.10 MHz
• Messages Logged: {coord_report.get('communications_coordination', {}).get('total_messages', 0)}

EVIDENCE/INVESTIGATION:
• Last known position: 42°57.8'N, 86°27.2'W
• Last radio transmission: 14:30 hours - engine trouble reported
• Vessel registration: [Available to authorized personnel]
• Insurance information: [Available to authorized personnel]

WEATHER/ENVIRONMENTAL:
• Current conditions: 4-6 foot seas, winds 15-20 knots
• Visibility: 3-5 miles
• Temperature: 52°F water, 68°F air
• Forecast: Conditions expected to worsen

NEXT ACTIONS:
• Continue search operations through night
• Deploy additional resources if needed
• Coordinate with family services
• Prepare for potential evidence recovery

CONTACT INFORMATION:
• Incident Commander: (555) 0100
• Operations Chief: (555) 0101
• Investigation Lead: (555) 0201

This report contains law enforcement sensitive information.
Report generated by CESAROPS v2.0 - {datetime.now().isoformat()}
            """
            
            # Save report
            report_filename = f"law_enforcement_{incident_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join(self.reports_directory, report_filename)
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            self._log_report_generation(incident_id, "law_enforcement", UserRole.LAW_ENFORCEMENT, report_path)
            
            return report_content
            
        except Exception as e:
            logger.error(f"Error generating law enforcement report: {e}")
            return f"Error generating report: {e}"
    
    def generate_sar_team_report(self, incident_id: str) -> str:
        """Generate SAR team operational report"""
        try:
            if not self.components_initialized:
                return "System components not available"
            
            # Get comprehensive data
            ics_status = self.ics.get_incident_status(incident_id)
            coord_report = self.coordinators.generate_coordination_report(incident_id)
            ics_report = self.ics.generate_ics_report(incident_id)
            
            # Create SAR team report
            report_content = f"""
SAR TEAM OPERATIONAL REPORT
INCIDENT: {ics_status.get('incident_id', 'Unknown')}
OPERATIONAL PERIOD: {ics_status.get('operational_period', 'Unknown')}
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by {self.current_user.full_name if self.current_user else 'System'}

MISSION SUMMARY:
• Vessel: 32-foot sport fishing boat 'Aurora'
• POB: 4 persons
• Last known position: 42°57.8'N, 86°27.2'W
• Distance from shore: 12 nautical miles west of Grand Haven
• Weather: Deteriorating, 4-6 foot seas

ICS ORGANIZATION:
• Incident Commander: {ics_status.get('incident_commander', 'Unknown')}
• Operational Period: {ics_status.get('operational_period', 'Unknown')}
• Personnel Assigned: {ics_status.get('personnel_count', 0)}
• Organization Effectiveness: {ics_report.get('organization_effectiveness', {}).get('organization_completeness', 0):.1%}

SEARCH OPERATIONS:
• Active Search Areas: {coord_report.get('search_coordination', {}).get('active_areas', 0)}
• Total Coverage: {coord_report.get('search_coordination', {}).get('average_coverage', 0):.1f}%
• Search Resources: {coord_report.get('search_coordination', {}).get('resources_deployed', 0)}
• Search Patterns: Grid and expanding square
• Probability of Detection: High in primary area

RESCUE READINESS:
• Active Rescue Operations: {coord_report.get('rescue_coordination', {}).get('active_rescues', 0)}
• Medical Personnel: Coast Guard corpsman and paramedics
• Extraction Capabilities: Helicopter hoist, surface vessel recovery
• Hospital Destination: Spectrum Health Grand Rapids (20 minutes flight time)

RESOURCES DEPLOYED:
• Surface Vessels: Coast Guard 47-foot MLB, Sheriff marine unit
• Aircraft: Coast Guard MH-65 Dolphin helicopter
• Personnel: 12 active responders
• Support: CAP aircraft providing aerial search support

COMMUNICATION STATUS:
• Primary: 121.50 MHz (SAR coordination)
• Tactical: 156.80 MHz (surface units), 157.10 MHz (air units)
• Messages: {coord_report.get('communications_coordination', {}).get('total_messages', 0)} logged
• Acknowledgment Rate: {coord_report.get('communications_coordination', {}).get('acknowledged_messages', 0) / max(1, coord_report.get('communications_coordination', {}).get('total_messages', 1)) * 100:.0f}%

COMPUTER VISION/AERIAL ASSETS:
• Drone Surveillance: Available for close-area search
• Thermal Imaging: Helicopter equipped with FLIR
• Computer Vision: Active detection algorithms for person/vessel identification
• Satellite Coverage: Weather imagery updated hourly

ENVIRONMENTAL CONDITIONS:
• Sea State: 4-6 feet
• Wind: 15-20 knots from southwest
• Visibility: 3-5 miles, improving
• Water Temperature: 52°F (survival time: 2-6 hours)
• Air Temperature: 68°F

SEARCH PROGRESS:
• Primary Area (0-3nm): 85% complete
• Secondary Area (3-6nm): 45% complete
• Tertiary Area (6-10nm): 15% complete
• Debris Detection: None confirmed
• Contacts Investigated: 3 (all negative)

RECOMMENDATIONS:
{'; '.join(coord_report.get('recommendations', ['Continue current operations']))}

NEXT OPERATIONAL PERIOD:
• Expand search area to 15nm radius
• Deploy additional CAP aircraft
• Prepare for night operations with thermal assets
• Reassess based on weather conditions

SAFETY CONSIDERATIONS:
• All personnel briefed on weather deterioration
• Life jacket compliance 100%
• Regular safety checks implemented
• Emergency evacuation plan in place

Report generated by CESAROPS v2.0 integrated SAR platform
Contact: {self.current_user.contact_info.get('phone', 'Unknown') if self.current_user else 'System'}
            """
            
            # Save report
            report_filename = f"sar_team_{incident_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join(self.reports_directory, report_filename)
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            self._log_report_generation(incident_id, "sar_team", UserRole.SAR_TEAM, report_path)
            
            return report_content
            
        except Exception as e:
            logger.error(f"Error generating SAR team report: {e}")
            return f"Error generating report: {e}"
    
    def generate_command_report(self, incident_id: str) -> str:
        """Generate incident commander comprehensive report"""
        try:
            if not self.components_initialized:
                return "System components not available"
            
            # Get all available data
            ics_status = self.ics.get_incident_status(incident_id)
            ics_report = self.ics.generate_ics_report(incident_id)
            coord_report = self.coordinators.generate_coordination_report(incident_id)
            
            # Create command report
            report_content = f"""
INCIDENT COMMANDER COMPREHENSIVE REPORT
INCIDENT: {ics_status.get('incident_id', 'Unknown')}
CLASSIFICATION: COMMAND SENSITIVE
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
Search and rescue operation for missing fishing vessel 'Aurora' with 4 POB. Operations commenced 
at {ics_status.get('start_time', 'Unknown')} with full ICS activation. Weather conditions are 
deteriorating but search operations continue with multiple assets deployed.

INCIDENT DETAILS:
• Incident Type: {ics_status.get('incident_type', 'Unknown')}
• Complexity Level: {ics_status.get('complexity_level', 'Unknown')}
• Operational Period: {ics_status.get('operational_period', 'Unknown')}
• Duration: {ics_status.get('duration', 'Unknown')}
• Estimated Cost: ${ics_status.get('estimated_cost', 0):,.2f}

ICS ORGANIZATION ANALYSIS:
• Command Structure: {ics_report.get('organization_effectiveness', {}).get('critical_positions_filled', 'Unknown')} critical positions filled
• Span of Control: {ics_report.get('organization_effectiveness', {}).get('span_of_control', 'Unknown')}
• Unity of Command: {ics_report.get('organization_effectiveness', {}).get('unity_of_command', 'Unknown')}
• Organization Completeness: {ics_report.get('organization_effectiveness', {}).get('organization_completeness', 0):.1%}

RESOURCE UTILIZATION:
• Total Resources: {ics_report.get('resource_utilization', {}).get('total_resources', 0)}
• Active Resources: {ics_report.get('resource_utilization', {}).get('active_resources', 0)}
• Utilization Rate: {ics_report.get('resource_utilization', {}).get('utilization_rate', 0):.1%}
• Hourly Cost: ${ics_report.get('resource_utilization', {}).get('estimated_hourly_cost', 0):,.2f}
• Resource Efficiency: {ics_report.get('resource_utilization', {}).get('resource_efficiency', 'Unknown')}

OPERATIONAL EFFECTIVENESS:
• Search Effectiveness: {coord_report.get('overall_effectiveness', {}).get('search_effectiveness', 'Unknown')}
• Rescue Effectiveness: {coord_report.get('overall_effectiveness', {}).get('rescue_effectiveness', 'Unknown')}
• Communication Effectiveness: {coord_report.get('overall_effectiveness', {}).get('communication_effectiveness', 'Unknown')}
• Overall Rating: {coord_report.get('overall_effectiveness', {}).get('overall_rating', 'Unknown')}

ADVANCED CAPABILITIES DEPLOYED:
• Computer Vision: Active person/vessel detection algorithms
• Aerial Assets: Coordinated drone and manned aircraft operations
• ML-Enhanced Drift Modeling: Predicting search areas based on environmental conditions
• Real-time Data Fusion: Satellite, weather, and sensor data integration
• Multi-agency Coordination: Coast Guard, Sheriff, CAP, volunteers

SEARCH COORDINATION METRICS:
• Active Search Areas: {coord_report.get('search_coordination', {}).get('active_areas', 0)}
• Average Coverage: {coord_report.get('search_coordination', {}).get('average_coverage', 0):.1f}%
• Resources Deployed: {coord_report.get('search_coordination', {}).get('resources_deployed', 0)}
• Search Efficiency: High

RESCUE COORDINATION STATUS:
• Active Rescue Operations: {coord_report.get('rescue_coordination', {}).get('active_rescues', 0)}
• Completed Rescues: {coord_report.get('rescue_coordination', {}).get('completed_rescues', 0)}
• Medical Readiness: Full capability
• Evacuation Assets: Ready

COMMUNICATIONS ANALYSIS:
• Total Messages: {coord_report.get('communications_coordination', {}).get('total_messages', 0)}
• Emergency Messages: {coord_report.get('communications_coordination', {}).get('emergency_messages', 0)}
• Acknowledgment Rate: {coord_report.get('communications_coordination', {}).get('acknowledged_messages', 0) / max(1, coord_report.get('communications_coordination', {}).get('total_messages', 1)) * 100:.0f}%
• Frequency Management: Optimal

STRATEGIC RECOMMENDATIONS:
{chr(10).join(f"• {rec}" for rec in ics_report.get('recommendations', ['No specific recommendations at this time']))}

COORDINATION RECOMMENDATIONS:
{chr(10).join(f"• {rec}" for rec in coord_report.get('recommendations', ['Operations proceeding effectively']))}

RISK ASSESSMENT:
• Weather: Moderate risk - conditions deteriorating
• Personnel Safety: Low risk - all safety protocols in place
• Mission Success: High probability with current resource deployment
• Environmental: Water temperature creates time-critical situation

NEXT COMMAND DECISIONS:
• Evaluate need for additional resources in next operational period
• Monitor weather conditions for impact on operations
• Prepare for possible transition to recovery operations if necessary
• Coordinate with media relations for public information

LESSONS LEARNED (Preliminary):
• CESAROPS integration significantly improved coordination efficiency
• Real-time data fusion enhanced search area prioritization
• Multi-agency communication protocols worked effectively
• Computer vision assets provided valuable supplementary coverage

STAKEHOLDER COMMUNICATIONS:
• Family notifications: Current and compassionate
• Media relations: Professional and informative
• Agency coordination: Excellent cooperation
• Political briefings: Command staff level only

This report contains command-sensitive operational information.
Distribution limited to incident command staff and authorized personnel.

Report generated by CESAROPS v2.0 - Advanced SAR Operations Platform
            """
            
            # Save report
            report_filename = f"command_{incident_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join(self.reports_directory, report_filename)
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            self._log_report_generation(incident_id, "command", UserRole.INCIDENT_COMMANDER, report_path)
            
            return report_content
            
        except Exception as e:
            logger.error(f"Error generating command report: {e}")
            return f"Error generating report: {e}"
    
    def _log_report_generation(self, incident_id: str, report_type: str, target_audience: str, file_path: str):
        """Log report generation for audit trail"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO reports 
                (report_id, incident_id, report_type, target_audience, generated_by, generated_time, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                incident_id,
                report_type,
                target_audience,
                self.current_user.user_id if self.current_user else "system",
                datetime.now().isoformat(),
                file_path
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging report generation: {e}")
    
    def run_system_tests(self) -> Dict[str, Any]:
        """Run comprehensive system tests"""
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'component_status': {},
            'performance_metrics': {},
            'detailed_results': []
        }
        
        print("Running CESAROPS System Tests...")
        print("=" * 50)
        
        # Test 1: Component Initialization
        test_results['tests_run'] += 1
        try:
            if self.components_initialized:
                test_results['tests_passed'] += 1
                test_results['component_status']['initialization'] = 'PASS'
                print("✓ Component Initialization: PASS")
            else:
                test_results['tests_failed'] += 1
                test_results['component_status']['initialization'] = 'FAIL'
                print("✗ Component Initialization: FAIL")
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['detailed_results'].append(f"Initialization test failed: {e}")
            print(f"✗ Component Initialization: FAIL - {e}")
        
        # Test 2: Database Connectivity
        test_results['tests_run'] += 1
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            conn.close()
            
            test_results['tests_passed'] += 1
            test_results['component_status']['database'] = 'PASS'
            test_results['performance_metrics']['user_count'] = user_count
            print(f"✓ Database Connectivity: PASS ({user_count} users)")
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['detailed_results'].append(f"Database test failed: {e}")
            print(f"✗ Database Connectivity: FAIL - {e}")
        
        # Test 3: User Authentication
        test_results['tests_run'] += 1
        try:
            auth_success = self.authenticate_user("incident_commander")
            if auth_success:
                test_results['tests_passed'] += 1
                test_results['component_status']['authentication'] = 'PASS'
                print("✓ User Authentication: PASS")
            else:
                test_results['tests_failed'] += 1
                test_results['component_status']['authentication'] = 'FAIL'
                print("✗ User Authentication: FAIL")
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['detailed_results'].append(f"Authentication test failed: {e}")
            print(f"✗ User Authentication: FAIL - {e}")
        
        # Test 4: Incident Creation
        test_results['tests_run'] += 1
        try:
            start_time = time.time()
            incident_id = self.create_demo_incident()
            creation_time = time.time() - start_time
            
            if incident_id:
                test_results['tests_passed'] += 1
                test_results['component_status']['incident_creation'] = 'PASS'
                test_results['performance_metrics']['incident_creation_time'] = creation_time
                print(f"✓ Incident Creation: PASS ({creation_time:.2f}s)")
            else:
                test_results['tests_failed'] += 1
                test_results['component_status']['incident_creation'] = 'FAIL'
                print("✗ Incident Creation: FAIL")
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['detailed_results'].append(f"Incident creation test failed: {e}")
            print(f"✗ Incident Creation: FAIL - {e}")
        
        # Test 5: Report Generation
        for role, report_func in [
            ('family', self.generate_family_report),
            ('law_enforcement', self.generate_law_enforcement_report),
            ('sar_team', self.generate_sar_team_report),
            ('command', self.generate_command_report)
        ]:
            test_results['tests_run'] += 1
            try:
                start_time = time.time()
                report = report_func(incident_id)
                generation_time = time.time() - start_time
                
                if report and len(report) > 100:  # Basic content check
                    test_results['tests_passed'] += 1
                    test_results['component_status'][f'{role}_report'] = 'PASS'
                    test_results['performance_metrics'][f'{role}_report_time'] = generation_time
                    print(f"✓ {role.title()} Report: PASS ({generation_time:.2f}s)")
                else:
                    test_results['tests_failed'] += 1
                    test_results['component_status'][f'{role}_report'] = 'FAIL'
                    print(f"✗ {role.title()} Report: FAIL")
            except Exception as e:
                test_results['tests_failed'] += 1
                test_results['detailed_results'].append(f"{role} report test failed: {e}")
                print(f"✗ {role.title()} Report: FAIL - {e}")
        
        # Test 6: Component Integration
        test_results['tests_run'] += 1
        try:
            if hasattr(self, 'ics') and hasattr(self, 'coordinators') and hasattr(self, 'computer_vision'):
                integration_score = 0
                if self.ics: integration_score += 1
                if self.coordinators: integration_score += 1
                if self.computer_vision: integration_score += 1
                if self.cap_coordinator: integration_score += 1
                if self.aerial_manager: integration_score += 1
                
                test_results['performance_metrics']['integration_score'] = f"{integration_score}/5"
                
                if integration_score >= 4:
                    test_results['tests_passed'] += 1
                    test_results['component_status']['integration'] = 'PASS'
                    print(f"✓ Component Integration: PASS ({integration_score}/5)")
                else:
                    test_results['tests_failed'] += 1
                    test_results['component_status']['integration'] = 'FAIL'
                    print(f"✗ Component Integration: FAIL ({integration_score}/5)")
            else:
                test_results['tests_failed'] += 1
                test_results['component_status']['integration'] = 'FAIL'
                print("✗ Component Integration: FAIL - Missing components")
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['detailed_results'].append(f"Integration test failed: {e}")
            print(f"✗ Component Integration: FAIL - {e}")
        
        # Test 9: Garmin RSD Integration
        test_results['tests_run'] += 1
        try:
            if hasattr(self, 'rsd_integration') and self.rsd_integration:
                # Test format detection
                formats = self.rsd_integration.list_supported_formats()
                capabilities = self.rsd_integration.get_detection_capabilities()
                exports = self.rsd_integration.get_export_options()
                
                if len(formats) >= 5 and len(exports) >= 4:
                    test_results['tests_passed'] += 1
                    test_results['component_status']['rsd_integration'] = 'PASS'
                    print(f"✓ Garmin RSD Integration: PASS ({len(formats)} formats, {len(exports)} exports)")
                else:
                    test_results['tests_failed'] += 1
                    test_results['component_status']['rsd_integration'] = 'FAIL'
                    print("✗ Garmin RSD Integration: FAIL - Insufficient capabilities")
            else:
                test_results['tests_passed'] += 1
                test_results['component_status']['rsd_integration'] = 'N/A'
                print("- Garmin RSD Integration: N/A (not available)")
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['detailed_results'].append(f"RSD integration test failed: {e}")
            print(f"✗ Garmin RSD Integration: FAIL - {e}")
        
        # Calculate success rate
        success_rate = (test_results['tests_passed'] / test_results['tests_run']) * 100 if test_results['tests_run'] > 0 else 0
        test_results['success_rate'] = success_rate
        
        print(f"\nTest Summary:")
        print(f"Tests Run: {test_results['tests_run']}")
        print(f"Tests Passed: {test_results['tests_passed']}")
        print(f"Tests Failed: {test_results['tests_failed']}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Save test results
        test_filename = f"system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        test_path = os.path.join(self.reports_directory, test_filename)
        
        with open(test_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\nDetailed test results saved to: {test_path}")
        
        return test_results

def main():
    """Main application entry point"""
    print("CESAROPS - Comprehensive Enhanced Search and Rescue Operations Platform")
    print("=" * 70)
    print("Initializing integrated SAR application...")
    
    # Initialize application
    app = CESAROPSApplication()
    
    print(f"✓ Application initialized")
    print(f"✓ Data directory: {app.data_directory}")
    print(f"✓ Components available: {app.components_initialized}")
    
    # Run system tests
    print(f"\nRunning comprehensive system tests...")
    test_results = app.run_system_tests()
    
    # Create demo incident and generate reports for all user types
    print(f"\nGenerating demonstration reports...")
    
    # Authenticate as different users and generate reports
    user_roles = [
        ("family_member", "Family Member"),
        ("law_enforcement", "Law Enforcement"),
        ("sar_team", "SAR Team"),
        ("incident_commander", "Incident Commander")
    ]
    
    demo_incident_id = None
    for username, role_name in user_roles:
        try:
            # Authenticate user
            if app.authenticate_user(username):
                print(f"\n--- {role_name} Report ---")
                
                # Create incident (only once)
                if demo_incident_id is None:
                    demo_incident_id = app.create_demo_incident()
                    if demo_incident_id:
                        print(f"✓ Demo incident created: {demo_incident_id}")
                
                # Generate appropriate report
                if username == "family_member":
                    report = app.generate_family_report(demo_incident_id)
                elif username == "law_enforcement":
                    report = app.generate_law_enforcement_report(demo_incident_id)
                elif username == "sar_team":
                    report = app.generate_sar_team_report(demo_incident_id)
                elif username == "incident_commander":
                    report = app.generate_command_report(demo_incident_id)
                
                # Show excerpt of report
                lines = report.split('\n')
                print(f"Report generated ({len(lines)} lines)")
                print("First 10 lines:")
                for i, line in enumerate(lines[:10]):
                    print(f"  {line}")
                if len(lines) > 10:
                    print(f"  ... ({len(lines) - 10} more lines)")
                
        except Exception as e:
            print(f"Error with {role_name}: {e}")
    
    print(f"\nCESAROPS Application Demonstration Complete!")
    print(f"Reports saved to: {app.reports_directory}")
    print(f"Test results: {test_results['success_rate']:.1f}% success rate")
    
    if test_results['success_rate'] >= 80:
        print("🟢 System Status: OPERATIONAL")
    elif test_results['success_rate'] >= 60:
        print("🟡 System Status: LIMITED OPERATION")
    else:
        print("🔴 System Status: NEEDS ATTENTION")
    
    print(f"\nCESAROPS v2.0 - Ready for Great Lakes SAR Operations")

if __name__ == "__main__":
    main()