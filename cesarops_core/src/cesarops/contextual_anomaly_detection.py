"""
Phase 13.3: Enhanced Anomaly Detection Module

Advanced anomaly detection with contextual analysis, pattern recognition,
and explainability for intelligent SAR operations.

Classes:
    RiskLevel: Enumeration of risk levels
    Explanation: Data class for anomaly explanations
    Pattern: Recognized anomaly pattern
    ContextualAnomalyDetector: Main detection engine
    AnomalyPatternLearner: ML-based pattern learning
    AnomalyExplainer: Human-readable explanation generation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Enumerations
# ============================================================================

class RiskLevel(Enum):
    """Risk assessment levels for anomalies."""
    LOW = "low"  # Known pattern, no SAR relevance
    MEDIUM = "medium"  # Partially understood
    HIGH = "high"  # Potential SAR target
    CRITICAL = "critical"  # Likely person or vessel


class AnomalyType(Enum):
    """Types of detected anomalies."""
    FISH_SCHOOL = "fish_school"
    GEOLOGICAL = "geological"
    MARINE_LIFE = "marine_life"
    ARTIFACT = "artifact"
    DEBRIS_FIELD = "debris_field"
    ANOMALOUS_RETURN = "anomalous_return"
    MULTIPATH = "multipath"
    THERMAL_NOISE = "thermal_noise"
    UNKNOWN = "unknown"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Pattern:
    """Recognized anomaly pattern."""
    pattern_id: str
    pattern_type: AnomalyType
    name: str
    description: str
    
    # Characteristics
    signal_characteristics: Dict[str, float]  # Frequency, intensity, duration, etc.
    environmental_factors: Dict[str, Tuple[float, float]]  # Expected ranges (min, max)
    
    # Context
    typical_depth_range: Tuple[float, float] = (0.0, 1000.0)
    typical_location: Optional[str] = None
    seasonal_factors: Dict[str, str] = field(default_factory=dict)
    
    # Confidence
    base_confidence: float = 0.8
    match_threshold: float = 0.7
    
    # SAR Relevance
    sar_relevant: bool = False
    typical_risk_level: RiskLevel = RiskLevel.MEDIUM
    
    def matches(self, characteristics: Dict[str, float]) -> Tuple[bool, float]:
        """
        Check if characteristics match this pattern.
        
        Returns:
            (matches: bool, confidence: float)
        """
        if not self.signal_characteristics:
            return False, 0.0
        
        matches = 0
        total = len(self.signal_characteristics)
        
        for key, expected_value in self.signal_characteristics.items():
            if key in characteristics:
                actual_value = characteristics[key]
                # Simple distance-based matching
                # Normalize difference relative to expected value
                if expected_value != 0:
                    diff = abs(actual_value - expected_value) / abs(expected_value)
                    if diff < 0.5:  # Within 50% tolerance
                        matches += 1
        
        confidence = (matches / total) * self.base_confidence if total > 0 else 0.0
        return matches > 0, confidence


@dataclass
class Explanation:
    """Human-readable explanation for an anomaly."""
    anomaly_id: str
    timestamp: float
    
    # Main conclusion
    conclusion: str  # What was detected
    reasoning: str  # Why it was flagged
    
    # Evidence
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    
    # Alternatives
    alternative_explanations: List[str] = field(default_factory=list)
    most_likely_cause: str = "unknown"
    
    # Confidence
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.5
    
    # Risk
    assessed_risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_justification: str = ""
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "timestamp": self.timestamp,
            "conclusion": self.conclusion,
            "reasoning": self.reasoning,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "alternative_explanations": self.alternative_explanations,
            "most_likely_cause": self.most_likely_cause,
            "confidence_factors": self.confidence_factors,
            "overall_confidence": self.overall_confidence,
            "risk_level": self.assessed_risk_level.value,
            "risk_justification": self.risk_justification,
            "recommended_actions": self.recommended_actions
        }
    
    def to_markdown(self) -> str:
        """Convert explanation to markdown report."""
        lines = [
            f"# Anomaly Report: {self.anomaly_id}",
            f"**Timestamp**: {datetime.fromtimestamp(self.timestamp).isoformat()}",
            "",
            "## Conclusion",
            self.conclusion,
            "",
            "## Reasoning",
            self.reasoning,
            "",
            "## Evidence",
        ]
        
        if self.supporting_evidence:
            lines.append("### Supporting Evidence")
            for evidence in self.supporting_evidence:
                lines.append(f"- {evidence}")
        
        if self.contradicting_evidence:
            lines.append("### Contradicting Evidence")
            for evidence in self.contradicting_evidence:
                lines.append(f"- {evidence}")
        
        if self.alternative_explanations:
            lines.append("## Alternative Explanations")
            for alt in self.alternative_explanations:
                lines.append(f"- {alt}")
        
        lines.extend([
            "",
            "## Confidence Assessment",
            f"**Overall Confidence**: {self.overall_confidence:.1%}",
            "",
            "### Confidence Factors",
        ])
        
        for factor, value in self.confidence_factors.items():
            lines.append(f"- {factor}: {value:.1%}")
        
        lines.extend([
            "",
            "## Risk Assessment",
            f"**Risk Level**: {self.assessed_risk_level.value.upper()}",
            f"**Justification**: {self.risk_justification}",
            "",
            "## Recommended Actions"
        ])
        
        if self.recommended_actions:
            for action in self.recommended_actions:
                lines.append(f"1. {action}")
        else:
            lines.append("- Continue monitoring")
        
        return "\n".join(lines)


@dataclass
class ContextualAnomaly:
    """Anomaly with full contextual analysis."""
    anomaly_id: str
    location: "SARLocation"  # Assuming this exists in another module
    timestamp: float
    
    # Detection results
    signal_intensity: float  # 0.0-1.0
    detected_type: AnomalyType = AnomalyType.UNKNOWN
    classification_confidence: float = 0.5
    
    # Context
    depth: float = 0.0
    bathymetric_features: List[str] = field(default_factory=list)
    temperature: float = 0.0
    current_speed: float = 0.0
    
    # Analysis results
    pattern_match: Optional[Pattern] = None
    pattern_match_confidence: float = 0.0
    
    # Assessment
    risk_level: RiskLevel = RiskLevel.MEDIUM
    anomaly_confidence: float = 0.0
    
    # Explainability
    explanation: Optional[Explanation] = None
    
    # Metadata
    contributing_sensors: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_combined_confidence(self) -> float:
        """Calculate combined confidence from multiple factors."""
        # Weighted average of confidence metrics
        weights = {
            "classification": 0.4,
            "pattern_match": 0.3,
            "signal_intensity": 0.2,
            "contextual": 0.1
        }
        
        confidence = (
            self.classification_confidence * weights["classification"] +
            self.pattern_match_confidence * weights["pattern_match"] +
            self.signal_intensity * weights["signal_intensity"] +
            (1.0 - abs(self.depth - 50.0) / 100.0) * weights["contextual"]
        )
        
        return min(1.0, max(0.0, confidence))


# ============================================================================
# Anomaly Pattern Learning
# ============================================================================

class AnomalyPatternLearner:
    """Learn anomaly patterns from historical data."""
    
    def __init__(self, training_data_path: Optional[str] = None):
        self.patterns: Dict[str, Pattern] = {}
        self.historical_detections: List[ContextualAnomaly] = []
        self.training_data_path = training_data_path
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self) -> None:
        """Initialize standard known patterns."""
        # Fish school pattern
        self.patterns["fish_school"] = Pattern(
            pattern_id="fish_school",
            pattern_type=AnomalyType.FISH_SCHOOL,
            name="Fish School",
            description="School of fish with distinctive swim patterns",
            signal_characteristics={
                "dominant_frequency": 1200.0,
                "intensity": 0.6,
                "duration": 3.0
            },
            environmental_factors={
                "temperature": (5.0, 25.0),
                "current_speed": (0.0, 2.0)
            },
            typical_depth_range=(10.0, 200.0),
            base_confidence=0.85,
            sar_relevant=False,
            typical_risk_level=RiskLevel.LOW
        )
        
        # Geological feature pattern
        self.patterns["geological"] = Pattern(
            pattern_id="geological",
            pattern_type=AnomalyType.GEOLOGICAL,
            name="Geological Feature",
            description="Rock formation, ridge, or seabed discontinuity",
            signal_characteristics={
                "dominant_frequency": 2000.0,
                "intensity": 0.8,
                "spectral_consistency": 0.9
            },
            environmental_factors={
                "temperature": (0.0, 30.0),
                "current_speed": (0.0, 3.0)
            },
            typical_depth_range=(0.0, 300.0),
            base_confidence=0.88,
            sar_relevant=False,
            typical_risk_level=RiskLevel.LOW
        )
        
        # Marine life pattern
        self.patterns["marine_life"] = Pattern(
            pattern_id="marine_life",
            pattern_type=AnomalyType.MARINE_LIFE,
            name="Large Marine Life",
            description="Whale, large fish, or marine mammal",
            signal_characteristics={
                "dominant_frequency": 800.0,
                "intensity": 0.75,
                "movement_pattern": 0.5
            },
            environmental_factors={
                "temperature": (0.0, 25.0),
                "current_speed": (0.0, 2.5)
            },
            typical_depth_range=(5.0, 500.0),
            base_confidence=0.82,
            sar_relevant=True,
            typical_risk_level=RiskLevel.MEDIUM
        )
        
        # Artifact pattern
        self.patterns["artifact"] = Pattern(
            pattern_id="artifact",
            pattern_type=AnomalyType.ARTIFACT,
            name="Artifact/Wreck",
            description="Man-made object, wreck, or debris",
            signal_characteristics={
                "dominant_frequency": 3000.0,
                "intensity": 0.9,
                "shape_definition": 0.8
            },
            environmental_factors={
                "temperature": (0.0, 30.0),
                "current_speed": (0.0, 3.0)
            },
            typical_depth_range=(5.0, 400.0),
            base_confidence=0.90,
            sar_relevant=True,
            typical_risk_level=RiskLevel.HIGH
        )
        
        # Multipath noise pattern
        self.patterns["multipath"] = Pattern(
            pattern_id="multipath",
            pattern_type=AnomalyType.MULTIPATH,
            name="Multipath Noise",
            description="Echo, reflection, or multipath distortion",
            signal_characteristics={
                "spectral_spread": 0.8,
                "coherence": 0.3,
                "repeated_delays": 1.0
            },
            environmental_factors={
                "current_speed": (0.5, 5.0)
            },
            base_confidence=0.75,
            sar_relevant=False,
            typical_risk_level=RiskLevel.LOW
        )
    
    def add_pattern(self, pattern: Pattern) -> None:
        """Add custom pattern."""
        self.patterns[pattern.pattern_id] = pattern
    
    def learn_patterns(self) -> None:
        """Train pattern recognition from historical data."""
        if not self.historical_detections:
            return
        
        # Analyze historical detections to refine patterns
        for detection in self.historical_detections:
            if detection.pattern_match:
                # Update pattern statistics
                pattern = detection.pattern_match
                # In a real implementation, would update feature statistics
    
    def identify_known_pattern(self, anomaly: ContextualAnomaly) -> Optional[Tuple[Pattern, float]]:
        """
        Check if anomaly matches known pattern.
        
        Returns:
            (matching_pattern, confidence) or None
        """
        best_match: Optional[Pattern] = None
        best_confidence: float = 0.0
        
        characteristics = {
            "intensity": anomaly.signal_intensity,
            "depth": anomaly.depth,
            "temperature": anomaly.temperature,
            "current_speed": anomaly.current_speed
        }
        
        for pattern in self.patterns.values():
            matches, confidence = pattern.matches(characteristics)
            
            if matches and confidence > best_confidence:
                if confidence >= pattern.match_threshold:
                    best_match = pattern
                    best_confidence = confidence
        
        if best_match:
            return best_match, best_confidence
        
        return None
    
    def update_with_new_data(self, labeled_examples: List[Dict[str, Any]]) -> None:
        """Incrementally learn from validated examples."""
        for example in labeled_examples:
            # Extract pattern characteristics from labeled example
            # Update corresponding pattern or create new one
            anomaly_type = example.get("type", AnomalyType.UNKNOWN)
            characteristics = example.get("characteristics", {})
            
            # Find or create pattern for this type
            if anomaly_type.value in self.patterns:
                pattern = self.patterns[anomaly_type.value]
                # Update pattern with new data
                # (In real implementation, would use Bayesian updating or ML)
    
    def get_all_patterns(self) -> Dict[str, Pattern]:
        """Get all known patterns."""
        return self.patterns.copy()


# ============================================================================
# Anomaly Explainer
# ============================================================================

class AnomalyExplainer:
    """Generate human-readable explanations for anomalies."""
    
    def __init__(self, pattern_learner: AnomalyPatternLearner):
        self.pattern_learner = pattern_learner
        self.explanation_cache: Dict[str, Explanation] = {}
    
    def explain_detection(self, anomaly: ContextualAnomaly) -> Explanation:
        """Generate detailed explanation for anomaly."""
        
        # Start with base conclusion
        conclusion = self._generate_conclusion(anomaly)
        reasoning = self._generate_reasoning(anomaly)
        
        # Gather evidence
        supporting_evidence = self._gather_supporting_evidence(anomaly)
        contradicting_evidence = self._gather_contradicting_evidence(anomaly)
        alternatives = self._generate_alternative_explanations(anomaly)
        
        # Calculate confidence factors
        confidence_factors = self._calculate_confidence_factors(anomaly)
        overall_confidence = anomaly.get_combined_confidence()
        
        # Assess risk
        risk_level = self._assess_risk_level(anomaly, overall_confidence)
        risk_justification = self._justify_risk_level(anomaly, risk_level)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(anomaly, risk_level)
        
        explanation = Explanation(
            anomaly_id=anomaly.anomaly_id,
            timestamp=anomaly.timestamp,
            conclusion=conclusion,
            reasoning=reasoning,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            alternative_explanations=alternatives,
            most_likely_cause=anomaly.detected_type.value,
            confidence_factors=confidence_factors,
            overall_confidence=overall_confidence,
            assessed_risk_level=risk_level,
            risk_justification=risk_justification,
            recommended_actions=recommendations
        )
        
        # Cache explanation
        self.explanation_cache[anomaly.anomaly_id] = explanation
        
        return explanation
    
    def _generate_conclusion(self, anomaly: ContextualAnomaly) -> str:
        """Generate main conclusion."""
        type_name = anomaly.detected_type.value.replace("_", " ").title()
        confidence_pct = anomaly.classification_confidence * 100
        
        return (
            f"Detected {type_name} at depth {anomaly.depth:.1f}m "
            f"with {confidence_pct:.0f}% classification confidence."
        )
    
    def _generate_reasoning(self, anomaly: ContextualAnomaly) -> str:
        """Generate reasoning for detection."""
        factors = []
        
        if anomaly.signal_intensity > 0.7:
            factors.append(f"Strong signal intensity ({anomaly.signal_intensity:.2f})")
        
        if anomaly.classification_confidence > 0.8:
            factors.append("High classification confidence from signal analysis")
        
        if anomaly.pattern_match:
            factors.append(f"Matches known '{anomaly.pattern_match.name}' pattern")
        
        if anomaly.depth > 0:
            factors.append(f"Depth ({anomaly.depth:.1f}m) consistent with detection type")
        
        if factors:
            return "Detection was flagged based on: " + ", ".join(factors)
        return "Detection based on signal analysis and contextual assessment."
    
    def _gather_supporting_evidence(self, anomaly: ContextualAnomaly) -> List[str]:
        """Gather evidence supporting the detection."""
        evidence = []
        
        if anomaly.signal_intensity > 0.6:
            evidence.append(f"Signal intensity of {anomaly.signal_intensity:.2f} above noise floor")
        
        if anomaly.classification_confidence > 0.7:
            evidence.append(f"Signal classification confidence: {anomaly.classification_confidence:.1%}")
        
        if anomaly.pattern_match:
            evidence.append(
                f"Signal matches '{anomaly.pattern_match.name}' pattern "
                f"with {anomaly.pattern_match_confidence:.1%} confidence"
            )
        
        if anomaly.temperature > 0:
            evidence.append(f"Water temperature ({anomaly.temperature:.1f}°C) suitable for detection type")
        
        if anomaly.contributing_sensors:
            evidence.append(f"Confirmed by sensors: {', '.join(anomaly.contributing_sensors)}")
        
        return evidence
    
    def _gather_contradicting_evidence(self, anomaly: ContextualAnomaly) -> List[str]:
        """Gather contradicting evidence."""
        contradictions = []
        
        if anomaly.signal_intensity < 0.3:
            contradictions.append("Low signal intensity may indicate weak signal or noise")
        
        if anomaly.classification_confidence < 0.6:
            contradictions.append("Low classification confidence suggests ambiguous signal")
        
        if anomaly.current_speed > 2.0:
            contradictions.append("High current speed may distort signal characteristics")
        
        if anomaly.detected_type == AnomalyType.UNKNOWN:
            contradictions.append("Signal type could not be confidently classified")
        
        return contradictions
    
    def _generate_alternative_explanations(self, anomaly: ContextualAnomaly) -> List[str]:
        """Generate alternative explanations."""
        alternatives = []
        
        # Check for common false positives
        if anomaly.current_speed > 1.0:
            alternatives.append("Multipath distortion from strong current")
        
        if anomaly.temperature > 20.0:
            alternatives.append("Thermal noise from warm water layer")
        
        if anomaly.pattern_match_confidence < 0.6:
            alternatives.append("Unknown acoustic phenomenon")
        
        # Check for competing patterns
        other_patterns = self.pattern_learner.patterns.values()
        for pattern in other_patterns:
            if pattern.pattern_type != anomaly.detected_type:
                alternatives.append(f"Could be {pattern.name.lower()}")
        
        return alternatives[:3]  # Limit to top 3
    
    def _calculate_confidence_factors(self, anomaly: ContextualAnomaly) -> Dict[str, float]:
        """Calculate individual confidence factors."""
        return {
            "classification": anomaly.classification_confidence,
            "pattern_match": anomaly.pattern_match_confidence,
            "signal_strength": anomaly.signal_intensity,
            "sensor_quality": np.mean(list(anomaly.quality_metrics.values())) if anomaly.quality_metrics else 0.5,
            "contextual_fit": 0.7 if anomaly.depth > 0 else 0.5
        }
    
    def _assess_risk_level(self, anomaly: ContextualAnomaly, confidence: float) -> RiskLevel:
        """Assess risk level."""
        # Risk matrix: confidence × SAR relevance
        if anomaly.pattern_match and anomaly.pattern_match.sar_relevant:
            if confidence > 0.85:
                return RiskLevel.CRITICAL
            elif confidence > 0.70:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        elif anomaly.detected_type in [AnomalyType.ARTIFACT, AnomalyType.MARINE_LIFE]:
            if confidence > 0.80:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW if confidence > 0.75 else RiskLevel.MEDIUM
    
    def _justify_risk_level(self, anomaly: ContextualAnomaly, risk_level: RiskLevel) -> str:
        """Justify risk level assessment."""
        if risk_level == RiskLevel.CRITICAL:
            return "Very high confidence detection of SAR-relevant target."
        elif risk_level == RiskLevel.HIGH:
            return "Good confidence detection of potentially significant target."
        elif risk_level == RiskLevel.MEDIUM:
            return "Moderate confidence; requires further investigation."
        else:
            return "Low risk; likely natural feature or benign detection."
    
    def _generate_recommendations(self, anomaly: ContextualAnomaly, risk_level: RiskLevel) -> List[str]:
        """Generate recommended actions."""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("PRIORITY: Investigate immediately")
            recommendations.append("Alert SAR coordination center")
            recommendations.append("Dispatch survey vessel if available")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Schedule investigation within 1 hour")
            recommendations.append("Notify SAR team for potential response")
            recommendations.append("Collect additional sensor data")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Continue monitoring area")
            recommendations.append("Collect additional context data")
            recommendations.append("Update anomaly database for pattern learning")
        else:
            recommendations.append("Continue routine monitoring")
            recommendations.append("Log detection for statistical analysis")
        
        return recommendations
    
    def generate_report(self, anomaly: ContextualAnomaly) -> str:
        """Generate detailed analysis report."""
        if anomaly.anomaly_id not in self.explanation_cache:
            self.explain_detection(anomaly)
        
        explanation = self.explanation_cache[anomaly.anomaly_id]
        return explanation.to_markdown()
    
    def get_explanation(self, anomaly_id: str) -> Optional[Explanation]:
        """Retrieve cached explanation."""
        return self.explanation_cache.get(anomaly_id)


# ============================================================================
# Contextual Anomaly Detector
# ============================================================================

class ContextualAnomalyDetector:
    """Advanced anomaly detection with contextual analysis."""
    
    def __init__(self, classifier: Optional[Any] = None,
                 fusion_engine: Optional[Any] = None):
        self.classifier = classifier
        self.fusion_engine = fusion_engine
        self.pattern_learner = AnomalyPatternLearner()
        self.explainer = AnomalyExplainer(self.pattern_learner)
        self.detection_history: List[ContextualAnomaly] = []
        self.threshold = 0.5  # Base anomaly threshold
    
    def detect_anomalies(self, readings: Dict[str, Any],
                        location: "SARLocation") -> List[ContextualAnomaly]:
        """
        Detect anomalies considering full context.
        
        Returns list of detected anomalies with analysis.
        """
        anomalies = []
        
        # Extract sonar data
        sonar_intensity = readings.get("sonar_intensity", 0.0)
        
        # Check if above threshold
        if sonar_intensity <= self.threshold:
            return anomalies
        
        # Classify signal
        classification_type = AnomalyType.UNKNOWN
        classification_confidence = 0.5
        
        if self.classifier:
            # Use ML classifier
            result = self.classifier.classify(sonar_intensity)
            classification_type = result.get("type", AnomalyType.UNKNOWN)
            classification_confidence = result.get("confidence", 0.5)
        
        # Get contextual data
        depth = readings.get("depth", 0.0)
        temperature = readings.get("temperature", 0.0)
        current_speed = readings.get("current_speed", 0.0)
        
        # Create anomaly
        anomaly = ContextualAnomaly(
            anomaly_id=f"anomaly_{datetime.now().timestamp()}",
            location=location,
            timestamp=datetime.now().timestamp(),
            signal_intensity=sonar_intensity,
            detected_type=classification_type,
            classification_confidence=classification_confidence,
            depth=depth,
            temperature=temperature,
            current_speed=current_speed,
            contributing_sensors=list(readings.keys())
        )
        
        # Match against known patterns
        pattern_result = self.pattern_learner.identify_known_pattern(anomaly)
        if pattern_result:
            anomaly.pattern_match, anomaly.pattern_match_confidence = pattern_result
        
        # Generate explanation
        anomaly.explanation = self.explainer.explain_detection(anomaly)
        
        # Set final confidence
        anomaly.anomaly_confidence = anomaly.get_combined_confidence()
        anomaly.risk_level = anomaly.explanation.assessed_risk_level
        
        anomalies.append(anomaly)
        self.detection_history.append(anomaly)
        
        return anomalies
    
    def score_anomaly(self, anomaly: ContextualAnomaly) -> Dict[str, float]:
        """Generate confidence score with reasoning."""
        return {
            "overall_confidence": anomaly.anomaly_confidence,
            "classification_confidence": anomaly.classification_confidence,
            "pattern_match_confidence": anomaly.pattern_match_confidence,
            "combined_confidence": anomaly.get_combined_confidence(),
            "risk_score": self._risk_to_score(anomaly.risk_level)
        }
    
    def _risk_to_score(self, risk_level: RiskLevel) -> float:
        """Convert risk level to numeric score."""
        scores = {
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0
        }
        return scores.get(risk_level, 0.5)
    
    def get_anomaly_patterns(self) -> Dict[str, Pattern]:
        """Get learned anomaly patterns."""
        return self.pattern_learner.get_all_patterns()
    
    def get_detection_history(self, limit: int = 100) -> List[ContextualAnomaly]:
        """Get detection history."""
        return self.detection_history[-limit:]
    
    def get_high_risk_anomalies(self) -> List[ContextualAnomaly]:
        """Get only high-risk detections."""
        return [
            a for a in self.detection_history
            if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
    
    def update_threshold(self, new_threshold: float) -> None:
        """Adjust detection threshold."""
        self.threshold = max(0.0, min(1.0, new_threshold))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.detection_history:
            return {
                "total_detections": 0,
                "average_confidence": 0.0,
                "high_risk_count": 0
            }
        
        detections = self.detection_history
        high_risk = self.get_high_risk_anomalies()
        
        return {
            "total_detections": len(detections),
            "average_confidence": np.mean([d.anomaly_confidence for d in detections]),
            "high_risk_count": len(high_risk),
            "detection_rate_per_hour": len(detections) / 24.0,  # Simplified
            "pattern_match_rate": sum(1 for d in detections if d.pattern_match) / len(detections)
        }
