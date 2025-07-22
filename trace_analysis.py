import base64
import sys
import json
import re
import os
import sqlite3
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
from dateutil.parser import parse as parse_date
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from plotly.subplots import make_subplots
from jinja2 import Template
import nltk
import os

# Force download of required models in script mode (silent if already present)
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

# Ensure NLTK looks in the right directory
nltk.data.path.append(nltk_data_path)

# Enhanced keyword sets for better classification
SECURITY_KEYWORDS = {
    'authentication', 'authorization', 'token', 'encryption', 'login', 'security', 
    'hash', 'ssl', 'tls', 'certificate', 'password', 'secure', 'crypto', 'oauth',
    'session', 'cors', 'xss', 'csrf', 'injection', 'vulnerability', 'exploit',
    'firewall', 'audit', 'compliance', 'access', 'permission', 'role', 'privilege'
}

PRIVACY_KEYWORDS = {
    'gdpr', 'privacy', 'personal', 'data', 'consent', 'anonymize', 'patient', 
    'health', 'pii', 'sensitive', 'confidential', 'private', 'protected',
    'tracking', 'cookie', 'profile', 'behavioral', 'biometric', 'location',
    'medical', 'financial', 'demographic', 'opt-out', 'right-to-be-forgotten'
}

NOISE_PATTERNS = [
    r'^(fix|update|refactor|cleanup|minor|small|typo)',
    r'^\s*$',  # empty content
    r'^(wip|todo|fixme)\s*:?\s*$',  # incomplete markers
    r'^\d+\.\d+\.\d+$',  # version numbers only
    r'^merge\s+branch',  # merge commits
    r'^revert\s+',  # revert commits
]

@dataclass
class TraceLink:
    id: str
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    link_type: str
    content: str
    confidence: float
    security_relevance: float
    privacy_relevance: float
    timestamp: str
    metadata: Dict[str, Any]
    quality_score: float = 0.0
    semantic_similarity: float = 0.0
    is_noise: bool = False
    classification: str = ""

class TraceQualityAnalyzer:
    """WP1: Analysis and Characterization of Raw Traces"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.noise_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in NOISE_PATTERNS]
        
    def analyze_trace_characteristics(self, traces: List[TraceLink]) -> Dict[str, Any]:
        """Comprehensive analysis of raw trace characteristics"""
        analysis = {
            'total_traces': len(traces),
            'source_types': Counter([t.source_type for t in traces]),
            'link_types': Counter([t.link_type for t in traces]),
            'content_lengths': [len(t.content) for t in traces],
            'confidence_distribution': [t.confidence for t in traces],
            'timestamp_range': self._analyze_temporal_distribution(traces),
            'quality_issues': self._identify_quality_issues(traces),
            'content_patterns': self._analyze_content_patterns(traces),
            'metadata_analysis': self._analyze_metadata(traces)
        }
        return analysis
    
    def _analyze_temporal_distribution(self, traces: List[TraceLink]) -> Dict[str, Any]:
        """Analyze temporal patterns in traces"""
        timestamps = []
        for trace in traces:
            try:
                ts = parse_date(trace.timestamp)
                if ts.tzinfo is not None:
                    ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                else:
                    ts = ts.replace(tzinfo=None)
                timestamps.append(ts)
            except Exception as e:
                continue

        if not timestamps:
            return {'error': 'No valid timestamps found'}

        return {
            'earliest': min(timestamps).isoformat(),
            'latest': max(timestamps).isoformat(),
            'span_days': (max(timestamps) - min(timestamps)).days,
            'distribution_by_month': Counter([ts.strftime('%Y-%m') for ts in timestamps])
        }
    
    def _identify_quality_issues(self, traces: List[TraceLink]) -> Dict[str, Any]:
        """Identify common quality issues in traces"""
        issues = {
            'empty_content': 0,
            'very_short_content': 0,  # < 10 chars
            'very_long_content': 0,   # > 1000 chars
            'duplicate_content': 0,
            'missing_metadata': 0,
            'noise_patterns': 0,
            'low_confidence': 0,  # < 0.5
        }
        
        content_seen = set()
        for trace in traces:
            content = trace.content.strip()
            
            if not content:
                issues['empty_content'] += 1
            elif len(content) < 10:
                issues['very_short_content'] += 1
            elif len(content) > 1000:
                issues['very_long_content'] += 1
            
            if content in content_seen:
                issues['duplicate_content'] += 1
            else:
                content_seen.add(content)
            
            if not trace.metadata or len(trace.metadata) == 0:
                issues['missing_metadata'] += 1
            
            if any(pattern.match(content) for pattern in self.noise_patterns):
                issues['noise_patterns'] += 1
            
            if trace.confidence < 0.5:
                issues['low_confidence'] += 1
        
        return issues
    
    def _analyze_content_patterns(self, traces: List[TraceLink]) -> Dict[str, Any]:
        """Analyze patterns in trace content"""
        all_content = ' '.join([t.content for t in traces if t.content])
        words = word_tokenize(all_content.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        return {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / len(words) if words else 0,
            'most_common_words': Counter(words).most_common(20),
            'avg_content_length': np.mean([len(t.content) for t in traces]),
            'content_language_detection': self._detect_languages(traces)
        }
    
    def _detect_languages(self, traces: List[TraceLink]) -> Dict[str, int]:
        """Detect languages in trace content"""
        languages = Counter()
        for trace in traces:
            if len(trace.content) > 20:  # Only analyze substantial content
                try:
                    lang = TextBlob(trace.content).detect_language()
                    languages[lang] += 1
                except:
                    languages['unknown'] += 1
        return dict(languages)
    
    def _analyze_metadata(self, traces: List[TraceLink]) -> Dict[str, Any]:
        """Analyze metadata patterns"""
        metadata_keys = Counter()
        metadata_completeness = []
        
        for trace in traces:
            if trace.metadata:
                metadata_keys.update(trace.metadata.keys())
                metadata_completeness.append(len(trace.metadata))
            else:
                metadata_completeness.append(0)
        
        return {
            'common_metadata_keys': dict(metadata_keys.most_common(10)),
            'avg_metadata_completeness': np.mean(metadata_completeness),
            'metadata_coverage': len([t for t in traces if t.metadata]) / len(traces)
        }

class AdvancedTraceFilter:
    """WP2: Development of Filtering and Cleaning Techniques"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.noise_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in NOISE_PATTERNS]
        
    def clean_and_filter_traces(self, traces: List[TraceLink], 
                               min_confidence: float = 0.3,
                               min_quality_score: float = 0.2,
                               remove_duplicates: bool = True,
                               remove_noise: bool = True) -> List[TraceLink]:
        """Comprehensive trace cleaning and filtering"""
        
        # Step 1: Remove obvious noise
        if remove_noise:
            traces = self._remove_noise_traces(traces)
        
        # Step 2: Calculate advanced quality scores
        traces = self._calculate_quality_scores(traces)
        
        # Step 3: Detect and handle duplicates
        if remove_duplicates:
            traces = self._remove_duplicate_traces(traces)
        
        # Step 4: Apply confidence and quality filters
        filtered_traces = []
        for trace in traces:
            if (trace.confidence >= min_confidence and 
                trace.quality_score >= min_quality_score):
                filtered_traces.append(trace)
        
        # Step 5: Enhance remaining traces with semantic analysis
        filtered_traces = self._enhance_with_semantic_analysis(filtered_traces)
        
        return filtered_traces
    
    def _remove_noise_traces(self, traces: List[TraceLink]) -> List[TraceLink]:
        """Remove traces matching noise patterns"""
        clean_traces = []
        for trace in traces:
            is_noise = any(pattern.match(trace.content.strip()) for pattern in self.noise_patterns)
            trace.is_noise = is_noise
            if not is_noise:
                clean_traces.append(trace)
        return clean_traces
    
    def _calculate_quality_scores(self, traces: List[TraceLink]) -> List[TraceLink]:
        """Calculate comprehensive quality scores for traces"""
        for trace in traces:
            quality_factors = []
            
            # Content length factor (sweet spot: 20-200 chars)
            content_len = len(trace.content)
            if 20 <= content_len <= 200:
                quality_factors.append(1.0)
            elif content_len < 20:
                quality_factors.append(content_len / 20.0)
            else:
                quality_factors.append(max(0.3, 200 / content_len))
            
            # Metadata completeness
            metadata_score = min(1.0, len(trace.metadata) / 5.0) if trace.metadata else 0.0
            quality_factors.append(metadata_score)
            
            # Content complexity (word variety)
            words = word_tokenize(trace.content.lower())
            unique_words = set(words)
            complexity_score = min(1.0, len(unique_words) / max(1, len(words)))
            quality_factors.append(complexity_score)
            
            # Security/Privacy relevance bonus
            relevance_bonus = max(trace.security_relevance, trace.privacy_relevance)
            quality_factors.append(relevance_bonus)
            
            # Calculate weighted quality score
            trace.quality_score = np.mean(quality_factors)
        
        return traces
    
    def _remove_duplicate_traces(self, traces: List[TraceLink]) -> List[TraceLink]:
        """Remove duplicate traces based on content similarity"""
        if len(traces) < 2:
            return traces
        
        # Create content matrix for similarity calculation
        contents = [trace.content for trace in traces]
        try:
            tfidf_matrix = self.tfidf.fit_transform(contents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Use clustering to find duplicates
            clustering = DBSCAN(eps=0.8, min_samples=1, metric='precomputed')
            clusters = clustering.fit_predict(1 - similarity_matrix)
            
            # Keep only the highest quality trace from each cluster
            unique_traces = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in unique_traces or traces[i].quality_score > unique_traces[cluster_id].quality_score:
                    unique_traces[cluster_id] = traces[i]
            
            return list(unique_traces.values())
        except:
            # Fallback to simple duplicate removal
            seen_content = set()
            unique_traces = []
            for trace in traces:
                if trace.content not in seen_content:
                    seen_content.add(trace.content)
                    unique_traces.append(trace)
            return unique_traces
    
    def _enhance_with_semantic_analysis(self, traces: List[TraceLink]) -> List[TraceLink]:
        """Enhance traces with semantic similarity and classification"""
        if len(traces) < 2:
            return traces
        
        try:
            contents = [trace.content for trace in traces]
            tfidf_matrix = self.tfidf.fit_transform(contents)
            
            for i, trace in enumerate(traces):
                # Calculate average semantic similarity to other traces
                similarities = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix).flatten()
                trace.semantic_similarity = np.mean(similarities[similarities != 1.0])
                
                # Classify trace based on content and relevance
                trace.classification = self._classify_trace(trace)
        except:
            # Set default values if semantic analysis fails
            for trace in traces:
                trace.semantic_similarity = 0.5
                trace.classification = self._classify_trace(trace)
        
        return traces
    
    def _classify_trace(self, trace: TraceLink) -> str:
        """Classify trace based on security/privacy relevance"""
        if trace.security_relevance >= 0.7:
            return "HIGH_SECURITY"
        elif trace.privacy_relevance >= 0.7:
            return "HIGH_PRIVACY"
        elif trace.security_relevance >= 0.4:
            return "MEDIUM_SECURITY"
        elif trace.privacy_relevance >= 0.4:
            return "MEDIUM_PRIVACY"
        elif trace.quality_score >= 0.6:
            return "GENERAL_QUALITY"
        else:
            return "LOW_RELEVANCE"

class TraceEvaluator:
    """WP3: Evaluation of Filtering Effectiveness"""
    
    def __init__(self):
        self.evaluation_metrics = {}
    
    def evaluate_filtering_effectiveness(self, 
                                       original_traces: List[TraceLink], 
                                       filtered_traces: List[TraceLink],
                                       ground_truth: List[str] = None) -> Dict[str, Any]:
        """Comprehensive evaluation of filtering effectiveness"""
        
        evaluation = {
            'basic_metrics': self._calculate_basic_metrics(original_traces, filtered_traces),
            'quality_metrics': self._calculate_quality_metrics(original_traces, filtered_traces),
            'distribution_analysis': self._analyze_filter_distribution(original_traces, filtered_traces),
            'effectiveness_score': 0.0
        }
        
        if ground_truth:
            evaluation['ground_truth_metrics'] = self._evaluate_against_ground_truth(filtered_traces, ground_truth)
        
        # Calculate overall effectiveness score
        evaluation['effectiveness_score'] = self._calculate_effectiveness_score(evaluation)
        
        return evaluation
    
    def _calculate_basic_metrics(self, original: List[TraceLink], filtered: List[TraceLink]) -> Dict[str, Any]:
        """Calculate basic filtering metrics"""
        return {
            'original_count': len(original),
            'filtered_count': len(filtered),
            'reduction_ratio': 1 - (len(filtered) / len(original)) if original else 0,
            'avg_confidence_original': np.mean([t.confidence for t in original]) if original else 0,
            'avg_confidence_filtered': np.mean([t.confidence for t in filtered]) if filtered else 0,
            'avg_quality_original': np.mean([getattr(t, 'quality_score', 0) for t in original]) if original else 0,
            'avg_quality_filtered': np.mean([getattr(t, 'quality_score', 0) for t in filtered]) if filtered else 0,
        }
    
    def _calculate_quality_metrics(self, original: List[TraceLink], filtered: List[TraceLink]) -> Dict[str, Any]:
        """Calculate quality improvement metrics"""
        orig_security = [t.security_relevance for t in original]
        filt_security = [t.security_relevance for t in filtered]
        orig_privacy = [t.privacy_relevance for t in original]
        filt_privacy = [t.privacy_relevance for t in filtered]
        
        return {
            'security_relevance_improvement': (np.mean(filt_security) - np.mean(orig_security)) if orig_security and filt_security else 0,
            'privacy_relevance_improvement': (np.mean(filt_privacy) - np.mean(orig_privacy)) if orig_privacy and filt_privacy else 0,
            'high_quality_retention': len([t for t in filtered if getattr(t, 'quality_score', 0) > 0.7]) / len(filtered) if filtered else 0,
            'noise_removal_rate': len([t for t in original if getattr(t, 'is_noise', False)]) / len(original) if original else 0
        }
    
    def _analyze_filter_distribution(self, original: List[TraceLink], filtered: List[TraceLink]) -> Dict[str, Any]:
        """Analyze how filtering affects trace distribution"""
        orig_sources = Counter([t.source_type for t in original])
        filt_sources = Counter([t.source_type for t in filtered])
        
        orig_classifications = Counter([getattr(t, 'classification', 'UNKNOWN') for t in original])
        filt_classifications = Counter([getattr(t, 'classification', 'UNKNOWN') for t in filtered])
        
        return {
            'source_type_retention': {source: (filt_sources[source] / orig_sources[source]) if orig_sources[source] > 0 else 0 
                                    for source in orig_sources.keys()},
            'classification_distribution_original': dict(orig_classifications),
            'classification_distribution_filtered': dict(filt_classifications),
        }
    
    def _evaluate_against_ground_truth(self, filtered_traces: List[TraceLink], ground_truth: List[str]) -> Dict[str, Any]:
        """Evaluate against manually annotated ground truth"""
        filtered_ids = set([t.id for t in filtered_traces])
        ground_truth_set = set(ground_truth)
        
        true_positives = len(filtered_ids.intersection(ground_truth_set))
        false_positives = len(filtered_ids - ground_truth_set)
        false_negatives = len(ground_truth_set - filtered_ids)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _calculate_effectiveness_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall effectiveness score (0-1)"""
        basic = evaluation['basic_metrics']
        quality = evaluation['quality_metrics']
        
        # Weighted combination of various factors
        factors = []
        
        # Quality improvement factor
        factors.append(max(0, basic['avg_quality_filtered'] - basic['avg_quality_original']))
        
        # Confidence improvement factor
        factors.append(max(0, basic['avg_confidence_filtered'] - basic['avg_confidence_original']))
        
        # Relevance improvement factor
        factors.append(max(0, quality['security_relevance_improvement']))
        factors.append(max(0, quality['privacy_relevance_improvement']))
        
        # Noise removal factor
        factors.append(quality['noise_removal_rate'])
        
        # High quality retention factor
        factors.append(quality['high_quality_retention'])
        
        return np.mean(factors) if factors else 0.0

class EnhancedTraceExtractor:
    """Enhanced version of the original extractor with better analysis"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def _calculate_enhanced_security_relevance(self, text: str) -> float:
        """Enhanced security relevance calculation using TF-IDF-like scoring"""
        words = set(word_tokenize(text.lower()))
        matches = words.intersection(SECURITY_KEYWORDS)
        
        # Basic keyword matching
        basic_score = min(len(matches) / 3.0, 1.0)
        
        # Context-aware scoring
        context_score = 0.0
        security_phrases = ['security issue', 'vulnerability', 'authentication', 'encrypted', 'secure connection']
        text_lower = text.lower()
        for phrase in security_phrases:
            if phrase in text_lower:
                context_score += 0.2
        
        return min(1.0, basic_score + min(context_score, 0.5))

    def _calculate_enhanced_privacy_relevance(self, text: str) -> float:
        """Enhanced privacy relevance calculation"""
        words = set(word_tokenize(text.lower()))
        matches = words.intersection(PRIVACY_KEYWORDS)
        
        # Basic keyword matching
        basic_score = min(len(matches) / 3.0, 1.0)
        
        # Context-aware scoring
        context_score = 0.0
        privacy_phrases = ['personal data', 'user privacy', 'data protection', 'consent', 'anonymization']
        text_lower = text.lower()
        for phrase in privacy_phrases:
            if phrase in text_lower:
                context_score += 0.2
        
        return min(1.0, basic_score + min(context_score, 0.5))

    def extract_github_traces(self, repo_url: str = "gematik/E-Rezept-App-Android") -> List[TraceLink]:
        """Extract traces from GitHub with enhanced analysis"""
        all_traces = []
        
        # Extract issues
        all_traces.extend(self._extract_issue_traces(repo_url))
        
        # Extract commits  
        all_traces.extend(self._extract_commit_traces(repo_url))
        
        # Extract simulated code traces
        all_traces.extend(self._extract_code_traces())
        
        return all_traces
    
    def _extract_issue_traces(self, repo: str) -> List[TraceLink]:
        """Extract and analyze GitHub issues"""
        traces = []
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": "all", "per_page": 100}
        headers = {"Accept": "application/vnd.github.v3+json"}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            issues = response.json()

            for issue in issues:
                if "pull_request" in issue:
                    continue

                content = f"{issue.get('title', '')} {issue.get('body', '') or ''}"
                
                # Enhanced confidence calculation
                confidence = 0.9
                if not issue.get('body'):
                    confidence -= 0.2
                if len(issue.get('labels', [])) == 0:
                    confidence -= 0.1
                
                trace = TraceLink(
                    id=f"issue_{issue['number']}",
                    source_type="issue",
                    source_id=str(issue['number']),
                    target_type="requirement",
                    target_id="unknown",
                    link_type="addresses",
                    content=content,
                    confidence=max(0.1, confidence),
                    security_relevance=self._calculate_enhanced_security_relevance(content),
                    privacy_relevance=self._calculate_enhanced_privacy_relevance(content),
                    timestamp=issue.get("created_at", datetime.now().isoformat()),
                    metadata={
                        "labels": [label["name"] for label in issue.get("labels", [])], 
                        "state": issue.get("state", ""),
                        "comments": issue.get("comments", 0),
                        "assignee": issue.get("assignee", {}).get("login", "") if issue.get("assignee") else ""
                    }
                )
                traces.append(trace)
                
        except Exception as e:
            print(f"Error fetching GitHub issues: {e}")

        return traces
    
    def _extract_commit_traces(self, repo: str) -> List[TraceLink]:
        """Extract and analyze commit traces"""
        traces = []
        url = f"https://api.github.com/repos/{repo}/commits"
        params = {"per_page": 50}
        headers = {"Accept": "application/vnd.github.v3+json"}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            commits = response.json()

            for commit in commits:
                sha = commit["sha"]
                message = commit["commit"]["message"]
                author = commit["commit"]["author"].get("name", "unknown")
                date = commit["commit"]["author"].get("date", datetime.now().isoformat())

                # Enhanced confidence based on commit message quality
                confidence = 0.8
                if len(message) < 10:
                    confidence -= 0.3
                if re.match(r'^(merge|revert)', message.lower()):
                    confidence -= 0.2

                trace = TraceLink(
                    id=f"commit_{sha[:8]}",
                    source_type="commit",
                    source_id=sha,
                    target_type="code",
                    target_id="unknown",
                    link_type="modifies",
                    content=message,
                    confidence=max(0.1, confidence),
                    security_relevance=self._calculate_enhanced_security_relevance(message),
                    privacy_relevance=self._calculate_enhanced_privacy_relevance(message),
                    timestamp=date,
                    metadata={
                        "author": author,
                        "sha": sha,
                        "full_message": message
                    }
                )
                traces.append(trace)
                
        except Exception as e:
            print(f"Error fetching commits: {e}")

        return traces
    
    def _extract_code_traces(self) -> List[TraceLink]:
        """Extract simulated code traces with realistic examples"""
        example_traces = [
            {
                "file": "security/AuthManager.kt",
                "line": 45,
                "content": "// TODO: Implement biometric authentication for enhanced security",
                "type": "todo"
            },
            {
                "file": "privacy/DataHandler.kt", 
                "line": 88,
                "content": "// PRIVACY: Ensure GDPR compliance when storing patient data",
                "type": "privacy_comment"
            },
            {
                "file": "network/ApiClient.kt",
                "line": 123,
                "content": "// Security: Add certificate pinning for API calls",
                "type": "security_comment"
            },
            {
                "file": "storage/DatabaseManager.kt",
                "line": 67,
                "content": "// Encrypt sensitive health data before storage",
                "type": "security_requirement"
            },
            {
                "file": "ui/LoginFragment.kt",
                "line": 234,
                "content": "// Handle user consent for data processing according to GDPR",
                "type": "privacy_requirement"
            }
        ]

        traces = []
        for code in example_traces:
            content = code["content"]
            confidence = 0.7 if code["type"] in ["todo"] else 0.85
            
            trace = TraceLink(
                id=f"code_{hash(content)}",
                source_type="code_comment",
                source_id=f"{code['file']}:{code['line']}",
                target_type="implementation",
                target_id="unknown",
                link_type="documents",
                content=content,
                confidence=confidence,
                security_relevance=self._calculate_enhanced_security_relevance(content),
                privacy_relevance=self._calculate_enhanced_privacy_relevance(content),
                timestamp=datetime.now().isoformat(),
                metadata={
                    "file": code["file"], 
                    "line": code["line"],
                    "comment_type": code["type"]
                }
            )
            traces.append(trace)

        return traces

def create_comprehensive_analysis(repo_url: str = "gematik/E-Rezept-App-Android") -> Dict[str, Any]:
    """WP4: Complete analysis workflow"""
    
    print("🚀 Starting Comprehensive Trace Analysis")
    print("="*60)
    
    # Initialize components
    extractor = EnhancedTraceExtractor()
    analyzer = TraceQualityAnalyzer()
    filter_system = AdvancedTraceFilter()
    evaluator = TraceEvaluator()
    
    # Step 1: Extract raw traces
    print("📊 Step 1: Extracting raw traces...")
    raw_traces = extractor.extract_github_traces(repo_url)
    print(f"   Extracted {len(raw_traces)} raw traces")
    
    # Step 2: Analyze raw trace characteristics (WP1)
    print("🔍 Step 2: Analyzing trace characteristics...")
    characteristics = analyzer.analyze_trace_characteristics(raw_traces)
    print(f"   Found {characteristics['quality_issues']['noise_patterns']} noise patterns")
    print(f"   Identified {characteristics['quality_issues']['duplicate_content']} potential duplicates")
    
    # Step 3: Apply filtering and cleaning (WP2)
    print("🧹 Step 3: Filtering and cleaning traces...")
    filtered_traces = filter_system.clean_and_filter_traces(
        raw_traces,
        min_confidence=0.4,
        min_quality_score=0.3,
        remove_duplicates=True,
        remove_noise=True
    )
    print(f"   Filtered to {len(filtered_traces)} high-quality traces")
    
    # Step 4: Evaluate filtering effectiveness (WP3)
    print("📈 Step 4: Evaluating filtering effectiveness...")
    evaluation = evaluator.evaluate_filtering_effectiveness(raw_traces, filtered_traces)
    print(f"   Reduction ratio: {evaluation['basic_metrics']['reduction_ratio']:.2f}")
    print(f"   Quality improvement: {evaluation['basic_metrics']['avg_quality_filtered']:.2f}")
    print(f"   Effectiveness score: {evaluation['effectiveness_score']:.2f}")
    
    # Step 5: Generate comprehensive report
    print("📋 Step 5: Generating comprehensive report...")
    report = {
        'extraction_summary': {
            'total_raw_traces': len(raw_traces),
            'total_filtered_traces': len(filtered_traces),
            'extraction_timestamp': datetime.now().isoformat()
        },
        'trace_characteristics': characteristics,
        'filtering_results': {
            'filtered_traces': [asdict(trace) for trace in filtered_traces],
            'filtering_settings': {
                'min_confidence': 0.4,
                'min_quality_score': 0.3,
                'remove_duplicates': True,
                'remove_noise': True
            }
        },
        'evaluation_results': evaluation
    }
    
    return report, raw_traces, filtered_traces

def visualize_comprehensive_analysis(raw_traces: List[TraceLink], 
                                   filtered_traces: List[TraceLink], 
                                   characteristics: Dict[str, Any]) -> None:
    """Create comprehensive visualizations for all analysis results"""
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Comprehensive Trace Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # 1. Trace Count Comparison (2x4 grid, position 1)
    plt.subplot(4, 4, 1)
    categories = ['Raw Traces', 'Filtered Traces', 'High Security', 'High Privacy', 'Noise Removed']
    counts = [
        len(raw_traces),
        len(filtered_traces),
        len([t for t in filtered_traces if t.security_relevance >= 0.7]),
        len([t for t in filtered_traces if t.privacy_relevance >= 0.7]),
        len(raw_traces) - len(filtered_traces)
    ]
    colors = ['lightblue', 'lightgreen', 'orange', 'red', 'gray']
    plt.bar(categories, counts, color=colors)
    plt.title('Trace Processing Summary')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # 2. Quality Score Distribution (position 2)
    plt.subplot(4, 4, 2)
    raw_quality = [getattr(t, 'quality_score', 0) for t in raw_traces]
    filtered_quality = [t.quality_score for t in filtered_traces]
    plt.hist(raw_quality, bins=20, alpha=0.5, label='Raw', color='lightcoral')
    plt.hist(filtered_quality, bins=20, alpha=0.7, label='Filtered', color='lightgreen')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.title('Quality Score Distribution')
    plt.legend()
    
    # 3. Security vs Privacy Relevance (position 3)
    plt.subplot(4, 4, 3)
    sec_scores = [t.security_relevance for t in filtered_traces]
    priv_scores = [t.privacy_relevance for t in filtered_traces]
    plt.scatter(sec_scores, priv_scores, alpha=0.6, c='purple', s=50)
    plt.xlabel('Security Relevance')
    plt.ylabel('Privacy Relevance')
    plt.title('Security vs Privacy Relevance')
    plt.grid(True, alpha=0.3)
    
    # 4. Source Type Distribution (position 4)
    plt.subplot(4, 4, 4)
    source_counts = Counter([t.source_type for t in filtered_traces])
    plt.pie(source_counts.values(), labels=source_counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Source Type Distribution')
    
    # 5. Quality Issues Analysis (position 5)
    plt.subplot(4, 4, 5)
    if 'quality_issues' in characteristics:
        issues = characteristics['quality_issues']
        issue_names = list(issues.keys())
        issue_counts = list(issues.values())
        plt.barh(issue_names, issue_counts, color='salmon')
        plt.title('Quality Issues Identified')
        plt.xlabel('Count')
    
    # 6. Confidence Distribution Comparison (position 6)
    plt.subplot(4, 4, 6)
    raw_conf = [t.confidence for t in raw_traces]
    filt_conf = [t.confidence for t in filtered_traces]
    plt.boxplot([raw_conf, filt_conf], labels=['Raw', 'Filtered'])
    plt.title('Confidence Score Comparison')
    plt.ylabel('Confidence')
    
    # 7. Content Length Distribution (position 7)
    plt.subplot(4, 4, 7)
    content_lengths = [len(t.content) for t in filtered_traces]
    plt.hist(content_lengths, bins=30, color='skyblue', alpha=0.7)
    plt.xlabel('Content Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Content Length Distribution')
    
    # 8. Classification Distribution (position 8)
    plt.subplot(4, 4, 8)
    classifications = Counter([getattr(t, 'classification', 'UNKNOWN') for t in filtered_traces])
    plt.bar(classifications.keys(), classifications.values(), color='lightsteelblue')
    plt.title('Trace Classifications')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # 9. Temporal Distribution (position 9-10, spans 2 columns)
    plt.subplot(4, 4, (9, 10))
    timestamps = []
    for trace in filtered_traces:
        try:
            ts = datetime.fromisoformat(trace.timestamp.replace('Z', '+00:00'))
            timestamps.append(ts)
        except:
            continue
    
    if timestamps:
        # Group by month
        monthly_counts = Counter([ts.strftime('%Y-%m') for ts in timestamps])
        months = sorted(monthly_counts.keys())
        counts = [monthly_counts[month] for month in months]
        
        plt.plot(range(len(months)), counts, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Time Period')
        plt.ylabel('Trace Count')
        plt.title('Temporal Distribution of Traces')
        plt.xticks(range(0, len(months), max(1, len(months)//5)), 
                  months[::max(1, len(months)//5)], rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 10. Top Keywords Analysis (position 11-12, spans 2 columns)
    plt.subplot(4, 4, (11, 12))
    all_content = ' '.join([t.content for t in filtered_traces])
    words = word_tokenize(all_content.lower())
    # Filter out common words and short words
    meaningful_words = [word for word in words 
                       if word.isalpha() and len(word) > 3 and 
                       word not in stopwords.words('english')]
    
    if meaningful_words:
        word_freq = Counter(meaningful_words)
        top_words = word_freq.most_common(15)
        words, freqs = zip(*top_words)
        
        plt.barh(words, freqs, color='mediumseagreen')
        plt.xlabel('Frequency')
        plt.title('Most Common Keywords in Filtered Traces')
    
    # 11. Relevance Heatmap (position 13-14, spans 2 columns)
    plt.subplot(4, 4, (13, 14))
    # Create relevance matrix by source type
    source_types = list(set([t.source_type for t in filtered_traces]))
    relevance_matrix = []
    
    for source_type in source_types:
        traces_of_type = [t for t in filtered_traces if t.source_type == source_type]
        if traces_of_type:
            avg_security = np.mean([t.security_relevance for t in traces_of_type])
            avg_privacy = np.mean([t.privacy_relevance for t in traces_of_type])
            avg_quality = np.mean([t.quality_score for t in traces_of_type])
            relevance_matrix.append([avg_security, avg_privacy, avg_quality])
        else:
            relevance_matrix.append([0, 0, 0])
    
    if relevance_matrix:
        sns.heatmap(relevance_matrix, 
                   xticklabels=['Security', 'Privacy', 'Quality'],
                   yticklabels=source_types,
                   annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Average Relevance Scores by Source Type')
    
    # 12. Filtering Effectiveness Metrics (position 15-16, spans 2 columns)  
    plt.subplot(4, 4, (15, 16))
    
    # Calculate various metrics
    original_avg_quality = np.mean([getattr(t, 'quality_score', 0) for t in raw_traces])
    filtered_avg_quality = np.mean([t.quality_score for t in filtered_traces])
    original_avg_security = np.mean([t.security_relevance for t in raw_traces])
    filtered_avg_security = np.mean([t.security_relevance for t in filtered_traces])
    original_avg_privacy = np.mean([t.privacy_relevance for t in raw_traces])
    filtered_avg_privacy = np.mean([t.privacy_relevance for t in filtered_traces])
    
    metrics = ['Quality Score', 'Security Relevance', 'Privacy Relevance']
    original_values = [original_avg_quality, original_avg_security, original_avg_privacy]
    filtered_values = [filtered_avg_quality, filtered_avg_security, filtered_avg_privacy]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original', color='lightcoral', alpha=0.7)
    plt.bar(x + width/2, filtered_values, width, label='Filtered', color='lightgreen', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Average Score')
    plt.title('Filtering Effectiveness: Before vs After')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('comprehensive_trace_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Comprehensive analysis saved as 'comprehensive_trace_analysis.png'")

def export_analysis_results(report: Dict[str, Any], 
                          raw_traces: List[TraceLink], 
                          filtered_traces: List[TraceLink]) -> None:
    """Export all analysis results to various formats"""
    
    # Export to CSV
    raw_df = pd.DataFrame([asdict(trace) for trace in raw_traces])
    filtered_df = pd.DataFrame([asdict(trace) for trace in filtered_traces])
    
    raw_df['metadata'] = raw_df['metadata'].apply(json.dumps)
    filtered_df['metadata'] = filtered_df['metadata'].apply(json.dumps)
    
    raw_df.to_csv('raw_traces_analysis.csv', index=False)
    filtered_df.to_csv('filtered_traces_analysis.csv', index=False)
    
    # Export to SQLite
    conn = sqlite3.connect('comprehensive_trace_analysis.db')
    raw_df.to_sql('raw_traces', conn, if_exists='replace', index=False)
    filtered_df.to_sql('filtered_traces', conn, if_exists='replace', index=False)
    
    # Store analysis metadata
    analysis_meta_df = pd.DataFrame([{
        'analysis_date': datetime.now().isoformat(),
        'total_raw_traces': len(raw_traces),
        'total_filtered_traces': len(filtered_traces),
        'reduction_ratio': report['evaluation_results']['basic_metrics']['reduction_ratio'],
        'effectiveness_score': report['evaluation_results']['effectiveness_score'],
        'avg_quality_improvement': (
            report['evaluation_results']['basic_metrics']['avg_quality_filtered'] - 
            report['evaluation_results']['basic_metrics']['avg_quality_original']
        )
    }])
    analysis_meta_df.to_sql('analysis_metadata', conn, if_exists='replace', index=False)
    conn.close()
    
    # Export comprehensive report to JSON
    with open('comprehensive_trace_analysis.html', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("📁 Analysis results exported:")
    print("   - raw_traces_analysis.csv")
    print("   - filtered_traces_analysis.csv") 
    print("   - comprehensive_trace_analysis.db")
    print("   - comprehensive_trace_analysis_report.json")

# =============================================
# 1. AUTO-TAGGING GITHUB ISSUES IMPLEMENTATION
# =============================================

class GitHubIntegrator:
    """Class to handle GitHub integration for auto-tagging issues"""
    
    def __init__(self, repo_url: str, token: str = None):
        self.repo_url = repo_url
        self.token = token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token is required for integration")
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        # Extract owner and repo from URL
        parts = repo_url.split('/')
        if len(parts) < 2:
            raise ValueError("Invalid repo URL format. Expected 'owner/repo'")
        self.owner = parts[0]
        self.repo = parts[1]
    
    def tag_issues(self, traces: List[TraceLink]):
        """Tag GitHub issues based on trace classification"""
        # We are only interested in issue traces that are classified and have a source_id (issue number)
        issue_traces = [t for t in traces if t.source_type == 'issue' and t.classification]
        
        for trace in issue_traces:
            issue_number = trace.source_id
            labels_to_add = self._get_labels_from_classification(trace.classification)
            if labels_to_add:
                self._add_labels_to_issue(issue_number, labels_to_add)
    
    def _get_labels_from_classification(self, classification: str) -> List[str]:
        """Map classification to GitHub labels"""
        label_map = {
            "HIGH_SECURITY": ["security", "high-priority"],
            "HIGH_PRIVACY": ["privacy", "high-priority"],
            "MEDIUM_SECURITY": ["security", "medium-priority"],
            "MEDIUM_PRIVACY": ["privacy", "medium-priority"],
            "GENERAL_QUALITY": ["traceability"],
            "LOW_RELEVANCE": []  # Don't tag low relevance
        }
        return label_map.get(classification, [])
    
    def _add_labels_to_issue(self, issue_number: str, labels: List[str]):
        """Add labels to a GitHub issue"""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{issue_number}/labels"
        data = {"labels": labels}
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            print(f"   Added labels {labels} to issue #{issue_number}")
        except Exception as e:
            print(f"   Error adding labels to issue #{issue_number}: {e}")

# ===================================
# 2. HTML DASHBOARD IMPLEMENTATION
# ===================================

def generate_html_dashboard(report: Dict[str, Any], filtered_traces: List[TraceLink], image_path: str = 'comprehensive_trace_analysis.png'):
    """Generate an HTML dashboard report"""
    # Convert image to base64 for embedding
    with open(image_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Create a DataFrame for the filtered traces
    df = pd.DataFrame([asdict(t) for t in filtered_traces])
    
    # Create interactive visualizations
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Security vs Privacy Relevance', 
        'Trace Classifications',
        'Quality Score Distribution',
        'Temporal Distribution'
    ))
    
    # Security vs Privacy scatter
    sec_scores = [t.security_relevance for t in filtered_traces]
    priv_scores = [t.privacy_relevance for t in filtered_traces]
    fig.add_trace(
        go.Scatter(
            x=sec_scores, y=priv_scores, mode='markers', 
            marker=dict(size=8, color=df['quality_score'], colorscale='Viridis'),
            text=df['content'], hoverinfo='text'
        ),
        row=1, col=1
    )
    
    # Classification distribution
    classifications = Counter([getattr(t, 'classification', 'UNKNOWN') for t in filtered_traces])
    fig.add_trace(
        go.Bar(
            x=list(classifications.keys()), 
            y=list(classifications.values()),
            marker_color='lightsteelblue'
        ),
        row=1, col=2
    )
    
    # Quality score distribution
    fig.add_trace(
        go.Histogram(
            x=df['quality_score'], 
            nbinsx=20, 
            marker_color='skyblue'
        ),
        row=2, col=1
    )
    
    # Temporal distribution
    if 'timestamp_range' in report['trace_characteristics']:
        monthly_counts = report['trace_characteristics']['timestamp_range']['distribution_by_month']
        if monthly_counts:
            months = list(monthly_counts.keys())
            counts = list(monthly_counts.values())
            fig.add_trace(
                go.Scatter(
                    x=months, y=counts, mode='lines+markers'
                ),
                row=2, col=2
            )
    
    fig.update_layout(height=800, showlegend=False)
    plot_html = fig.to_html(full_html=False)
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comprehensive Trace Analysis Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ padding: 20px; }}
            .dashboard-img {{ max-width: 100%; }}
            .card {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="my-4">Comprehensive Trace Analysis Dashboard</h1>
            <p class="text-muted">Generated at: {datetime.now().isoformat()}</p>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3>Summary Metrics</h3>
                        </div>
                        <div class="card-body">
                            <ul class="list-group">
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Total Raw Traces
                                    <span class="badge bg-primary rounded-pill">{report['extraction_summary']['total_raw_traces']}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Total Filtered Traces
                                    <span class="badge bg-success rounded-pill">{report['extraction_summary']['total_filtered_traces']}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Reduction Ratio
                                    <span class="badge bg-info rounded-pill">{report['evaluation_results']['basic_metrics']['reduction_ratio']:.2%}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Effectiveness Score
                                    <span class="badge bg-warning rounded-pill">{report['evaluation_results']['effectiveness_score']:.2f}/1.0</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3>Quality Improvements</h3>
                        </div>
                        <div class="card-body">
                            <ul class="list-group">
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Avg Quality (Original)
                                    <span class="badge bg-secondary rounded-pill">{report['evaluation_results']['basic_metrics']['avg_quality_original']:.2f}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Avg Quality (Filtered)
                                    <span class="badge bg-success rounded-pill">{report['evaluation_results']['basic_metrics']['avg_quality_filtered']:.2f}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Security Improvement
                                    <span class="badge bg-danger rounded-pill">{report['evaluation_results']['quality_metrics']['security_relevance_improvement']:.2f}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Privacy Improvement
                                    <span class="badge bg-primary rounded-pill">{report['evaluation_results']['quality_metrics']['privacy_relevance_improvement']:.2f}</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card mt-4">
                <div class="card-header">
                    <h3>Visual Dashboard</h3>
                </div>
                <div class="card-body">
                    {plot_html}
                </div>
            </div>
            
            <div class="card mt-4">
                <div class="card-header">
                    <h3>Filtered Traces</h3>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Source</th>
                                    <th>Content</th>
                                    <th>Security</th>
                                    <th>Privacy</th>
                                    <th>Classification</th>
                                </tr>
                            </thead>
                            <tbody>
                                {"".join([
                                    f"<tr><td>{t.id}</td><td>{t.source_type}</td><td>{t.content[:80]}...</td>"
                                    f"<td>{t.security_relevance:.2f}</td><td>{t.privacy_relevance:.2f}</td>"
                                    f"<td>{t.classification}</td></tr>" 
                                    for t in filtered_traces
                                ])}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open('comprehensive_trace_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("📊 HTML dashboard saved as 'comprehensive_trace_analysis.html'")

# ===================================
# 3. REGRESSION DETECTION
# ===================================

class RegressionDetector:
    """Detect regressions in trace links by comparing with a previous state"""
    
    def __init__(self, previous_state_path: str = 'previous_state.json'):
        self.previous_state_path = previous_state_path
        self.previous_traces = self._load_previous_state()
    
    def _load_previous_state(self) -> Dict[str, Dict]:
        """Load previous state from a JSON file"""
        if not os.path.exists(self.previous_state_path):
            return {}
        try:
            with open(self.previous_state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # Convert to a dictionary by trace id
            return {trace['id']: trace for trace in state}
        except:
            return {}
    
    def save_current_state(self, traces: List[TraceLink]):
        """Save the current state as the new previous state"""
        state_data = [asdict(t) for t in traces]
        with open(self.previous_state_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2)
    
    def detect_regressions(self, current_traces: List[TraceLink]) -> Dict[str, Any]:
        """Detect regressions by comparing current traces with previous state"""
        current_by_id = {t.id: t for t in current_traces}
        regressions = []
        
        for trace_id, prev_trace in self.previous_traces.items():
            # We only care about high security and privacy traces
            if prev_trace['classification'] in ['HIGH_SECURITY', 'HIGH_PRIVACY']:
                if trace_id not in current_by_id:
                    # This trace is missing in the current set
                    regressions.append({
                        'id': trace_id,
                        'previous_classification': prev_trace['classification'],
                        'content': prev_trace['content'],
                        'reason': 'Missing in current state'
                    })
                else:
                    # Check if the classification has been downgraded
                    current_trace = current_by_id[trace_id]
                    if current_trace.classification != prev_trace['classification']:
                        if (prev_trace['classification'] == 'HIGH_SECURITY' and 
                            current_trace.classification not in ['HIGH_SECURITY', 'MEDIUM_SECURITY']):
                            regressions.append({
                                'id': trace_id,
                                'previous_classification': prev_trace['classification'],
                                'current_classification': current_trace.classification,
                                'content': current_trace.content,
                                'reason': 'Downgraded classification'
                            })
                        elif (prev_trace['classification'] == 'HIGH_PRIVACY' and 
                            current_trace.classification not in ['HIGH_PRIVACY', 'MEDIUM_PRIVACY']):
                            regressions.append({
                                'id': trace_id,
                                'previous_classification': prev_trace['classification'],
                                'current_classification': current_trace.classification,
                                'content': current_trace.content,
                                'reason': 'Downgraded classification'
                            })
        
        return regressions

# ===================================
# 4. CONCEPT PAPER GENERATION
# ===================================

def generate_concept_paper(report: Dict[str, Any]):
    """Generate a concept paper in markdown format"""
    content = f"""
# Concept Paper: Automated Traceability for Security and Privacy Compliance

## Overview
This document outlines the concept and implementation of an automated traceability tool for security and privacy compliance in software development. The tool integrates into development environments to automatically extract, filter, and classify trace links from various artifacts (issues, commits, code comments) and provides mechanisms for compliance reporting and CI integration.

## Key Use Cases

### 1. Security Regression Check in CI
The tool can be integrated into the CI pipeline to check for missing or changed security trace links. If a commit or pull request removes or downgrades a high-security trace link, the build can be failed or a warning generated.

### 2. Auto-Tagging Issues
The tool automatically classifies issues as 'security', 'privacy', etc. based on their content and adds appropriate labels in the issue tracking system for easier triage.

### 3. Traceability Reports for Audits
The tool generates scheduled traceability reports in multiple formats (CSV, JSON, HTML) for compliance audits. These reports include dashboards and metrics on the traceability coverage for security and privacy requirements.

## Implementation Summary

### Extraction
The tool extracts trace links from:
- GitHub issues
- Commit messages
- Code comments (simulated)

### Filtering and Classification
A multi-stage filtering process removes noise and duplicates. Traces are classified based on their relevance to security and privacy using keyword matching and heuristics.

### Integration
The tool can be triggered via GitHub Actions. It outputs:
- CSV reports
- HTML dashboards
- JSON files for further processing

## Results
In the latest analysis of the repository `gematik/E-Rezept-App-Android`:
- **Raw traces extracted**: {report['extraction_summary']['total_raw_traces']}
- **High-quality traces retained**: {report['extraction_summary']['total_filtered_traces']}
- **Noise reduction**: {report['evaluation_results']['basic_metrics']['reduction_ratio']:.1%}
- **Effectiveness score**: {report['evaluation_results']['effectiveness_score']:.2f}

## Next Steps
- Integrate with more artifact sources (requirements, design documents)
- Improve classification with machine learning
- Support more issue trackers and CI systems

---
Generated by Traceability Tool on {datetime.now().isoformat()}
"""
    with open('concept_paper.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("📝 Concept paper saved as 'concept_paper.md'")

# ===================================
# UPDATED MAIN FUNCTION
# ===================================

def main():
    """Main execution function - WP4: Complete automated workflow"""
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("📦 Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    
    # Run comprehensive analysis
    print("Starting analysis...")
    repo_url = "gematik/E-Rezept-App-Android"
    report, raw_traces, filtered_traces = create_comprehensive_analysis(repo_url)
    print("Analysis completed successfully!")
    
    # Create visualizations
    visualize_comprehensive_analysis(raw_traces, filtered_traces, report['trace_characteristics'])
    
    # Generate HTML dashboard
    generate_html_dashboard(report, filtered_traces)
    
    # Export results
    export_analysis_results(report, raw_traces, filtered_traces)
    
    # Generate concept paper
    generate_concept_paper(report)
    
    # Auto-tag GitHub issues if token is available
    if os.getenv('GITHUB_TOKEN'):
        try:
            print("🏷️ Auto-tagging GitHub issues...")
            integrator = GitHubIntegrator(repo_url)
            integrator.tag_issues(filtered_traces)
        except Exception as e:
            print(f"⚠️ Failed to auto-tag issues: {e}")
    else:
        print("ℹ️ GitHub token not found. Skipping auto-tagging.")
    
    # Regression detection
    detector = RegressionDetector()
    regressions = detector.detect_regressions(filtered_traces)
    detector.save_current_state(filtered_traces)
    
    if regressions:
        print(f"⛔ Regression detected: {len(regressions)} high-security or high-privacy traces missing or downgraded!")
        for reg in regressions:
            print(f"   - {reg['id']}: {reg['reason']} (Previous: {reg['previous_classification']})")
        # If running in CI, we want to fail the build
        if os.getenv('CI'):
            print("##[error] Regression in trace links detected! Build failed.")
            sys.exit(1)
    else:
        print("✅ No regressions detected in high-security and high-privacy traces.")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE TRACE ANALYSIS COMPLETE")
    print("="*60)
    
    basic_metrics = report['evaluation_results']['basic_metrics']
    quality_metrics = report['evaluation_results']['quality_metrics']
    
    print(f"📊 SUMMARY STATISTICS:")
    print(f"   Raw traces extracted: {basic_metrics['original_count']}")
    print(f"   High-quality traces retained: {basic_metrics['filtered_count']}")
    print(f"   Noise reduction: {basic_metrics['reduction_ratio']:.1%}")
    print(f"   Quality improvement: {(basic_metrics['avg_quality_filtered'] - basic_metrics['avg_quality_original']):.2f}")
    print(f"   Security relevance improvement: {quality_metrics['security_relevance_improvement']:.2f}")
    print(f"   Privacy relevance improvement: {quality_metrics['privacy_relevance_improvement']:.2f}")
    print(f"   Overall effectiveness score: {report['evaluation_results']['effectiveness_score']:.2f}/1.0")
    
    print(f"\n🏆 WORK PACKAGE COMPLETION:")
    print(f"   ✅ WP1: Raw trace analysis and characterization - COMPLETE")
    print(f"   ✅ WP2: Advanced filtering and cleaning techniques - COMPLETE")  
    print(f"   ✅ WP3: Comprehensive evaluation framework - COMPLETE")
    print(f"   ✅ WP4: Automated workflow and integration - COMPLETE")
    
    print(f"\n🔍 RESEARCH QUESTION ADDRESSED:")
    print(f"   'How can raw traces be cleaned and filtered to retain plausible and meaningful trace links only?'")
    print(f"   → Implemented multi-stage filtering with {basic_metrics['reduction_ratio']:.1%} noise reduction")
    print(f"   → Achieved {report['evaluation_results']['effectiveness_score']:.2f}/1.0 effectiveness score")
    print(f"   → Retained {quality_metrics['high_quality_retention']:.1%} high-quality traces")
    
    return report, raw_traces, filtered_traces

if __name__ == "__main__":
    try:
        report, raw_traces, filtered_traces = main()
        print("Script completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()