#!/usr/bin/env python3
"""
Canadian Identity Survey Comprehensive Analysis Suite.
Runs all analyses and generates comprehensive insights dashboard.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import re
from datetime import datetime
from io import BytesIO, StringIO
import base64
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_validate_data(data_dir):
    """Load and validate all Canadian Identity Survey datasets."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"📁 Loading Canadian Identity Survey data from: {data_dir}")
    
    datasets = {}
    
    # Expected file patterns for Canadian Identity Survey
    file_patterns = {
        'survey_data': '*Canada Survey Data.csv',
        'numeric_data': '*Numeric Data.csv',
        'demographic_data': '*demographic data.csv',
        'data_dictionary': '*data dictionary.csv'
    }
    
    for dataset_name, pattern in file_patterns.items():
        files = list(data_dir.glob(pattern))
        if files:
            file_path = files[0]
            try:
                print(f"   Loading {dataset_name}: {file_path.name}")
                df = pd.read_csv(file_path, encoding='utf-8')
                datasets[dataset_name] = df
                print(f"   ✓ {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"   ⚠️ Error loading {dataset_name}: {e}")
                datasets[dataset_name] = None
        else:
            print(f"   ⚠️ No file found for {dataset_name} (pattern: {pattern})")
            datasets[dataset_name] = None
    
    if datasets['survey_data'] is None:
        raise ValueError("Could not load main survey data file")
    
    return datasets

def extract_survey_responses(datasets):
    """Extract structured survey responses from datasets."""
    survey_df = datasets['survey_data']
    
    print(f"   Extracting responses from {len(survey_df)} survey records")
    
    responses = []
    for idx, row in survey_df.iterrows():
        # Skip completely empty rows
        if pd.isna(row).all():
            continue
        
        # Extract participant ID
        participant_id = str(row.get('participant_id', idx)) if not pd.isna(row.get('participant_id')) else str(idx)
        
        # Structure the response data
        response = {
            'id': participant_id,
            'metadata': {
                'distribution_type': str(row.get('distribution_type', '')) if not pd.isna(row.get('distribution_type')) else '',
                'engagement_status': str(row.get('engagement_participation_status', '')) if not pd.isna(row.get('engagement_participation_status')) else '',
                'broker_panel_id': str(row.get('BrokerPanelId', '')) if not pd.isna(row.get('BrokerPanelId')) else ''
            },
            'demographics': {
                'province': str(row.get('Q1_province', '')) if not pd.isna(row.get('Q1_province')) else '',
                'urban_rural': str(row.get('Q1a_urban_rural', '')) if not pd.isna(row.get('Q1a_urban_rural')) else '',
                'canadian_tenure': str(row.get('Q2_canadian_tenure', '')) if not pd.isna(row.get('Q2_canadian_tenure')) else '',
                'generation': str(row.get('Q3_generation', '')) if not pd.isna(row.get('Q3_generation')) else ''
            },
            'responses': {
                'identity_connection': str(row.get('Q5_identity_connection', '')) if not pd.isna(row.get('Q5_identity_connection')) else '',
                'connection_why': str(row.get('Q5b_identity_connection_why', '')) if not pd.isna(row.get('Q5b_identity_connection_why')) else '',
                'one_word_canadian': str(row.get('Q6_oneword', '')) if not pd.isna(row.get('Q6_oneword')) else '',
                'future_aspiration': str(row.get('Q8_onesentance', '')) if not pd.isna(row.get('Q8_onesentance')) else '',
                'story_tagline': str(row.get('Q11_cover_tailgate', '')) if not pd.isna(row.get('Q11_cover_tailgate')) else '',
                'future_hope': str(row.get('Q12_Future_hope', '')) if not pd.isna(row.get('Q12_Future_hope')) else '',
                'future_fear': str(row.get('Q13_Future_fear', '')) if not pd.isna(row.get('Q13_Future_fear')) else ''
            }
        }
        
        # Add identity factors (Q4)
        identity_factors = {}
        for i in range(1, 9):
            col_name = f'Q4_identity_top3_{i}'
            if col_name in survey_df.columns:
                identity_factors[f'factor_{i}'] = str(row.get(col_name, '')) if not pd.isna(row.get(col_name)) else ''
        response['identity_factors'] = identity_factors
        
        # Add values (Q7)
        values = {}
        for i in range(1, 9):
            col_name = f'Q7_values_{i}'
            if col_name in survey_df.columns:
                values[f'value_{i}'] = str(row.get(col_name, '')) if not pd.isna(row.get(col_name)) else ''
        response['values'] = values
        
        # Add divisions (Q9)
        divisions = {}
        for i in range(1, 8):
            col_name = f'Q9_divisions_{i}'
            if col_name in survey_df.columns:
                divisions[f'division_{i}'] = str(row.get(col_name, '')) if not pd.isna(row.get(col_name)) else ''
        response['divisions'] = divisions
        
        # Add connection building (Q10)
        connections = {}
        for i in range(1, 8):
            col_name = f'Q10_Connection_building_{i}'
            if col_name in survey_df.columns:
                connections[f'connection_{i}'] = str(row.get(col_name, '')) if not pd.isna(row.get(col_name)) else ''
        response['connection_building'] = connections
        
        responses.append(response)
    
    return responses

def analyze_response_metadata(responses):
    """Analyze metadata about responses."""
    distribution_types = Counter(r['metadata']['distribution_type'] for r in responses if r['metadata']['distribution_type'])
    engagement_statuses = Counter(r['metadata']['engagement_status'] for r in responses if r['metadata']['engagement_status'])
    
    return {
        'total_responses': len(responses),
        'distribution_types': dict(distribution_types),
        'engagement_statuses': dict(engagement_statuses),
        'response_completeness': calculate_response_completeness(responses)
    }

def calculate_response_completeness(responses):
    """Calculate how complete responses are."""
    completeness_scores = []
    
    for response in responses:
        # Count non-empty responses across all categories
        total_fields = 0
        completed_fields = 0
        
        for category in ['demographics', 'responses', 'identity_factors', 'values', 'divisions', 'connection_building']:
            if category in response:
                for field, value in response[category].items():
                    total_fields += 1
                    if value and value.strip() and value.lower() not in ['nan', 'none', '']:
                        completed_fields += 1
        
        completeness = completed_fields / total_fields if total_fields > 0 else 0
        completeness_scores.append(completeness)
    
    return {
        'average_completeness': np.mean(completeness_scores),
        'completeness_distribution': {
            'high_completeness': sum(1 for score in completeness_scores if score > 0.8),
            'medium_completeness': sum(1 for score in completeness_scores if 0.5 <= score <= 0.8),
            'low_completeness': sum(1 for score in completeness_scores if score < 0.5)
        }
    }

def comprehensive_theme_analysis(responses):
    """Comprehensive analysis of themes across all responses."""
    themes = {
        'unity': ['unity', 'together', 'united', 'cohesion', 'harmony', 'solidarity', 'collective', 'common'],
        'diversity': ['diversity', 'multicultural', 'inclusion', 'diverse', 'different', 'variety', 'pluralism', 'tolerance'],
        'heritage': ['heritage', 'history', 'tradition', 'culture', 'ancestry', 'roots', 'background', 'legacy'],
        'values': ['freedom', 'democracy', 'equality', 'justice', 'fairness', 'rights', 'liberty', 'respect'],
        'geography': ['land', 'nature', 'landscape', 'beautiful', 'environment', 'wilderness', 'vast', 'territory'],
        'opportunity': ['opportunity', 'future', 'potential', 'growth', 'development', 'progress', 'advancement'],
        'community': ['community', 'people', 'neighbors', 'society', 'belonging', 'family', 'connection', 'support'],
        'challenges': ['challenge', 'problem', 'difficult', 'struggle', 'concern', 'issue', 'barrier', 'obstacle'],
        'government': ['government', 'political', 'policy', 'leadership', 'federal', 'provincial', 'civic', 'public'],
        'economy': ['economy', 'economic', 'jobs', 'employment', 'business', 'financial', 'prosperity', 'wealth'],
        'healthcare': ['healthcare', 'health', 'medical', 'care', 'hospital', 'system', 'universal', 'accessible'],
        'education': ['education', 'learning', 'school', 'knowledge', 'skills', 'training', 'university', 'literacy'],
        'reconciliation': ['reconciliation', 'indigenous', 'first nations', 'truth', 'healing', 'treaty', 'aboriginal'],
        'climate': ['climate', 'environment', 'green', 'sustainable', 'clean', 'emissions', 'carbon', 'renewable']
    }
    
    theme_scores = defaultdict(int)
    theme_sentiment = defaultdict(list)
    question_theme_matrix = defaultdict(lambda: defaultdict(int))
    
    for response in responses:
        # Analyze all text responses
        all_texts = {}
        all_texts.update(response['responses'])
        
        for question, answer in all_texts.items():
            if not answer or answer.lower().strip() in ['nan', 'none', '']:
                continue
            
            answer_lower = answer.lower()
            
            for theme, keywords in themes.items():
                matches = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
                if matches > 0:
                    theme_scores[theme] += matches
                    question_theme_matrix[question][theme] += matches
                    
                    # Simple sentiment analysis
                    positive_words = ['great', 'amazing', 'excellent', 'love', 'wonderful', 'fantastic', 'proud', 'excited', 'optimistic', 'hope', 'positive', 'beautiful']
                    negative_words = ['terrible', 'awful', 'hate', 'worried', 'concerned', 'disappointed', 'frustrated', 'angry', 'scared', 'fear', 'negative', 'bad']
                    
                    pos_count = sum(1 for word in positive_words if word.lower() in answer_lower)
                    neg_count = sum(1 for word in negative_words if word.lower() in answer_lower)
                    
                    if pos_count > neg_count:
                        sentiment = 'positive'
                    elif neg_count > pos_count:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    
                    theme_sentiment[theme].append(sentiment)
    
    # Calculate sentiment scores
    theme_sentiment_summary = {}
    for theme, sentiments in theme_sentiment.items():
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        theme_sentiment_summary[theme] = {
            'positive': sentiment_counts.get('positive', 0) / total,
            'negative': sentiment_counts.get('negative', 0) / total,
            'neutral': sentiment_counts.get('neutral', 0) / total
        } if total > 0 else {'positive': 0, 'negative': 0, 'neutral': 0}
    
    return {
        'theme_scores': dict(theme_scores),
        'theme_sentiment': theme_sentiment_summary,
        'question_theme_matrix': dict(question_theme_matrix)
    }

def analyze_demographic_patterns(responses):
    """Analyze demographic patterns and distributions."""
    demographic_analysis = {}
    
    # Province distribution
    provinces = Counter(r['demographics']['province'] for r in responses if r['demographics']['province'])
    demographic_analysis['provinces'] = dict(provinces)
    
    # Urban/Rural distribution
    urban_rural = Counter(r['demographics']['urban_rural'] for r in responses if r['demographics']['urban_rural'])
    demographic_analysis['urban_rural'] = dict(urban_rural)
    
    # Generation distribution
    generations = Counter(r['demographics']['generation'] for r in responses if r['demographics']['generation'])
    demographic_analysis['generations'] = dict(generations)
    
    # Canadian tenure
    tenure = Counter(r['demographics']['canadian_tenure'] for r in responses if r['demographics']['canadian_tenure'])
    demographic_analysis['tenure'] = dict(tenure)
    
    # Calculate demographic diversity metrics
    demographic_analysis['diversity_metrics'] = calculate_demographic_diversity(demographic_analysis)
    
    return demographic_analysis

def calculate_demographic_diversity(demographic_data):
    """Calculate diversity metrics for demographics."""
    diversity_metrics = {}
    
    for category, distribution in demographic_data.items():
        if isinstance(distribution, dict) and distribution:
            total = sum(distribution.values())
            if total > 0:
                # Shannon diversity index
                proportions = [count/total for count in distribution.values()]
                shannon_diversity = -sum(p * np.log(p) for p in proportions if p > 0)
                diversity_metrics[f'{category}_diversity'] = shannon_diversity
                
                # Representation balance (1 = perfectly balanced, 0 = completely unbalanced)
                max_possible_diversity = np.log(len(distribution))
                balance_score = shannon_diversity / max_possible_diversity if max_possible_diversity > 0 else 0
                diversity_metrics[f'{category}_balance'] = balance_score
    
    return diversity_metrics

def analyze_identity_connection(responses):
    """Analyze connection to Canadian identity."""
    identity_analysis = {}
    
    # Connection levels
    connection_levels = Counter(r['responses']['identity_connection'] for r in responses if r['responses']['identity_connection'])
    identity_analysis['connection_levels'] = dict(connection_levels)
    
    # Calculate connection strength score
    connection_mapping = {
        'Not at all connected': 1,
        'Not very connected': 2,
        'A little connected': 3,
        'Somewhat connected': 4,
        'Very connected': 5
    }
    
    connection_scores = []
    for response in responses:
        connection_text = response['responses']['identity_connection']
        if connection_text in connection_mapping:
            connection_scores.append(connection_mapping[connection_text])
    
    if connection_scores:
        identity_analysis['average_connection_score'] = np.mean(connection_scores)
        identity_analysis['connection_strength'] = categorize_connection_strength(np.mean(connection_scores))
    
    # Analyze one-word responses
    one_words = [r['responses']['one_word_canadian'] for r in responses if r['responses']['one_word_canadian']]
    identity_analysis['one_word_analysis'] = analyze_one_word_responses(one_words)
    
    # Analyze connection reasons
    connection_reasons = [r['responses']['connection_why'] for r in responses if r['responses']['connection_why']]
    identity_analysis['connection_themes'] = extract_connection_themes(connection_reasons)
    
    return identity_analysis

def categorize_connection_strength(score):
    """Categorize connection strength based on average score."""
    if score >= 4.5:
        return "Very Strong"
    elif score >= 3.5:
        return "Strong"
    elif score >= 2.5:
        return "Moderate"
    elif score >= 1.5:
        return "Weak"
    else:
        return "Very Weak"

def analyze_one_word_responses(one_words):
    """Analyze one-word responses about Canada."""
    word_counts = Counter()
    sentiment_categories = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    # Common positive and negative words
    positive_words = ['home', 'beautiful', 'free', 'diverse', 'great', 'peaceful', 'kind', 'welcoming', 'inclusive', 'strong', 'proud']
    negative_words = ['cold', 'expensive', 'divided', 'struggling', 'disappointing', 'polarized', 'broken']
    
    for word in one_words:
        if not word or word.lower().strip() in ['nan', 'none', '']:
            continue
        
        word_clean = str(word).strip().lower()
        word_counts[word_clean] += 1
        
        # Simple sentiment categorization
        if any(pos_word in word_clean for pos_word in positive_words):
            sentiment_categories['positive'] += 1
        elif any(neg_word in word_clean for neg_word in negative_words):
            sentiment_categories['negative'] += 1
        else:
            sentiment_categories['neutral'] += 1
    
    return {
        'top_words': dict(word_counts.most_common(20)),
        'sentiment_distribution': sentiment_categories,
        'total_responses': len([w for w in one_words if w and w.strip()])
    }

def extract_connection_themes(connection_reasons):
    """Extract themes from connection reasons."""
    themes = {
        'heritage': ['heritage', 'ancestry', 'family', 'roots', 'background', 'culture', 'tradition', 'born'],
        'values': ['values', 'freedom', 'democracy', 'equality', 'rights', 'justice', 'fair', 'principles'],
        'belonging': ['belong', 'home', 'welcome', 'accepted', 'community', 'part of', 'included', 'feel'],
        'pride': ['proud', 'pride', 'accomplishment', 'achievement', 'success', 'great', 'strong'],
        'geography': ['land', 'nature', 'landscape', 'beautiful', 'country', 'place', 'environment'],
        'opportunity': ['opportunity', 'chance', 'future', 'potential', 'growth', 'development', 'better'],
        'diversity': ['diverse', 'multicultural', 'different', 'variety', 'inclusive', 'tolerance', 'acceptance']
    }
    
    theme_counts = defaultdict(int)
    
    for reason in connection_reasons:
        if not reason or reason.lower().strip() in ['nan', 'none', '']:
            continue
        
        reason_lower = str(reason).lower()
        
        for theme, keywords in themes.items():
            if any(keyword in reason_lower for keyword in keywords):
                theme_counts[theme] += 1
    
    return {
        'theme_counts': dict(theme_counts),
        'total_analyzed': len([r for r in connection_reasons if r and r.strip()])
    }

def analyze_values_priorities(responses):
    """Analyze values priorities for Canada's future."""
    values_mapping = {
        'value_1': 'Equitable systems',
        'value_2': 'Freedom and autonomy',
        'value_3': 'Embracing pluralism',
        'value_4': 'Indigenous reconciliation and rights',
        'value_5': 'Environmental regeneration',
        'value_6': 'Technological innovation and adaptation',
        'value_7': 'Community care networks',
        'value_8': 'Other'
    }
    
    values_counts = Counter()
    
    for response in responses:
        for value_key, value_label in values_mapping.items():
            if response['values'].get(value_key, '').strip():
                values_counts[value_label] += 1
    
    # Calculate percentages
    total_responses = len(responses)
    values_percentages = {k: (v/total_responses)*100 for k, v in values_counts.items()}
    
    return {
        'values_counts': dict(values_counts),
        'values_percentages': values_percentages,
        'top_values': values_counts.most_common(5)
    }

def analyze_social_divisions(responses):
    """Analyze perceptions of social divisions."""
    divisions_mapping = {
        'division_1': 'Resource and regional tensions',
        'division_2': 'Political polarization',
        'division_3': 'Urban/rural disconnection',
        'division_4': 'Generational perspectives',
        'division_5': 'Economic inequality',
        'division_6': 'Cultural misunderstanding',
        'division_7': 'Other'
    }
    
    divisions_counts = Counter()
    
    for response in responses:
        for division_key, division_label in divisions_mapping.items():
            if response['divisions'].get(division_key, '').strip():
                divisions_counts[division_label] += 1
    
    # Calculate concern levels
    total_responses = len(responses)
    concern_percentages = {k: (v/total_responses)*100 for k, v in divisions_counts.items()}
    
    return {
        'divisions_counts': dict(divisions_counts),
        'concern_percentages': concern_percentages,
        'top_concerns': divisions_counts.most_common(3)
    }

def analyze_connection_building(responses):
    """Analyze strategies for building stronger connections."""
    connections_mapping = {
        'connection_1': 'Honest dialogue and listening',
        'connection_2': 'Collaborative community projects',
        'connection_3': 'Cultural celebration and exchange',
        'connection_4': 'Educational transformation',
        'connection_5': 'Indigenous-led reconciliation',
        'connection_6': 'Newcomer integration and knowledge',
        'connection_7': 'Other'
    }
    
    connections_counts = Counter()
    
    for response in responses:
        for connection_key, connection_label in connections_mapping.items():
            if response['connection_building'].get(connection_key, '').strip():
                connections_counts[connection_label] += 1
    
    # Calculate strategy preferences
    total_responses = len(responses)
    strategy_percentages = {k: (v/total_responses)*100 for k, v in connections_counts.items()}
    
    return {
        'strategies_counts': dict(connections_counts),
        'strategy_percentages': strategy_percentages,
        'top_strategies': connections_counts.most_common(3)
    }

def analyze_future_aspirations(responses):
    """Analyze hopes, fears, and aspirations for Canada's future."""
    future_analysis = {}
    
    # Analyze hopes
    hopes = [r['responses']['future_hope'] for r in responses if r['responses']['future_hope']]
    future_analysis['hopes'] = analyze_future_texts(hopes, 'hopes')
    
    # Analyze fears
    fears = [r['responses']['future_fear'] for r in responses if r['responses']['future_fear']]
    future_analysis['fears'] = analyze_future_texts(fears, 'fears')
    
    # Analyze aspirations
    aspirations = [r['responses']['future_aspiration'] for r in responses if r['responses']['future_aspiration']]
    future_analysis['aspirations'] = analyze_future_texts(aspirations, 'aspirations')
    
    return future_analysis

def analyze_future_texts(texts, text_type):
    """Analyze sentiment and themes in future-focused texts."""
    themes = {
        'unity': ['unity', 'together', 'united', 'cohesion', 'harmony'],
        'equality': ['equality', 'equal', 'fairness', 'justice', 'equity'],
        'prosperity': ['prosperity', 'growth', 'economy', 'wealth', 'success'],
        'environment': ['environment', 'climate', 'green', 'sustainable', 'clean'],
        'healthcare': ['healthcare', 'health', 'medical', 'care'],
        'education': ['education', 'learning', 'school', 'knowledge'],
        'technology': ['technology', 'innovation', 'digital', 'future'],
        'democracy': ['democracy', 'freedom', 'rights', 'liberty'],
        'diversity': ['diversity', 'multicultural', 'inclusion'],
        'reconciliation': ['reconciliation', 'indigenous', 'truth']
    }
    
    if text_type == 'fears':
        themes.update({
            'division': ['division', 'polarization', 'split', 'conflict'],
            'inequality': ['inequality', 'gap', 'disparity', 'unfair'],
            'decline': ['decline', 'collapse', 'failure', 'crisis'],
            'extremism': ['extremism', 'radical', 'violence', 'hate']
        })
    
    theme_counts = defaultdict(int)
    word_frequencies = Counter()
    
    for text in texts:
        if not text or text.lower().strip() in ['nan', 'none', '']:
            continue
        
        text_lower = str(text).lower()
        
        # Theme analysis
        for theme, keywords in themes.items():
            if any(keyword in text_lower for keyword in keywords):
                theme_counts[theme] += 1
        
        # Word frequency analysis
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        stop_words = {'the', 'and', 'for', 'are', 'with', 'that', 'will', 'have', 'can', 'would', 'could', 'should', 'this', 'but', 'not', 'all', 'our', 'more'}
        meaningful_words = [w for w in words if w not in stop_words]
        word_frequencies.update(meaningful_words)
    
    return {
        'theme_counts': dict(theme_counts),
        'top_words': dict(word_frequencies.most_common(15)),
        'total_responses': len([t for t in texts if t and t.strip()])
    }

def extract_actionable_insights(responses):
    """Extract actionable insights from all responses."""
    insights = {
        'identity_strengthening': [],
        'unity_building': [],
        'value_alignment': [],
        'division_addressing': [],
        'future_planning': []
    }
    
    # Analyze connection patterns
    strong_connections = [r for r in responses if r['responses']['identity_connection'] == 'Very connected']
    weak_connections = [r for r in responses if r['responses']['identity_connection'] in ['Not at all connected', 'Not very connected']]
    
    insights['connection_patterns'] = {
        'strong_connection_count': len(strong_connections),
        'weak_connection_count': len(weak_connections),
        'connection_ratio': len(strong_connections) / len(responses) if responses else 0
    }
    
    # Extract specific actionable items from text responses
    all_texts = []
    for response in responses:
        all_texts.extend([
            response['responses']['connection_why'],
            response['responses']['future_aspiration'],
            response['responses']['future_hope']
        ])
    
    action_patterns = extract_action_patterns(all_texts)
    insights['action_patterns'] = action_patterns
    
    return insights

def extract_action_patterns(texts):
    """Extract action patterns from text responses."""
    action_words = ['need', 'should', 'must', 'require', 'want', 'hope', 'wish', 'improve', 'change', 'create', 'build', 'develop']
    action_patterns = Counter()
    
    for text in texts:
        if not text or text.lower().strip() in ['nan', 'none', '']:
            continue
        
        text_lower = str(text).lower()
        for action in action_words:
            if action in text_lower:
                # Extract context around action word
                words = text_lower.split()
                for i, word in enumerate(words):
                    if action in word:
                        context = ' '.join(words[max(0, i-1):min(len(words), i+3)])
                        action_patterns[context] += 1
                        break
    
    return dict(action_patterns.most_common(10))

def generate_visualizations(analysis_results):
    """Generate visualization charts for the analysis."""
    visualizations = {}
    
    try:
        # 1. Demographics Overview
        plt.figure(figsize=(16, 12))
        
        # Province distribution
        if 'demographic_patterns' in analysis_results and 'provinces' in analysis_results['demographic_patterns']:
            plt.subplot(2, 3, 1)
            provinces = analysis_results['demographic_patterns']['provinces']
            sorted_provinces = sorted(provinces.items(), key=lambda x: x[1], reverse=True)[:8]
            plt.bar(range(len(sorted_provinces)), [x[1] for x in sorted_provinces])
            plt.xticks(range(len(sorted_provinces)), [x[0] for x in sorted_provinces], rotation=45)
            plt.title('Provincial Distribution')
            plt.ylabel('Count')
        
        # Urban/Rural distribution
        if 'demographic_patterns' in analysis_results and 'urban_rural' in analysis_results['demographic_patterns']:
            plt.subplot(2, 3, 2)
            urban_rural = analysis_results['demographic_patterns']['urban_rural']
            plt.pie(urban_rural.values(), labels=urban_rural.keys(), autopct='%1.1f%%')
            plt.title('Urban vs Rural Distribution')
        
        # Generation distribution
        if 'demographic_patterns' in analysis_results and 'generations' in analysis_results['demographic_patterns']:
            plt.subplot(2, 3, 3)
            generations = analysis_results['demographic_patterns']['generations']
            plt.bar(range(len(generations)), list(generations.values()))
            plt.xticks(range(len(generations)), list(generations.keys()), rotation=45)
            plt.title('Generational Distribution')
            plt.ylabel('Count')
        
        # Identity connection levels
        if 'identity_connection' in analysis_results and 'connection_levels' in analysis_results['identity_connection']:
            plt.subplot(2, 3, 4)
            connections = analysis_results['identity_connection']['connection_levels']
            plt.pie(connections.values(), labels=connections.keys(), autopct='%1.1f%%')
            plt.title('Canadian Identity Connection')
        
        # Top values
        if 'values_priorities' in analysis_results and 'values_percentages' in analysis_results['values_priorities']:
            plt.subplot(2, 3, 5)
            values = analysis_results['values_priorities']['values_percentages']
            sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)[:6]
            plt.barh(range(len(sorted_values)), [x[1] for x in sorted_values])
            plt.yticks(range(len(sorted_values)), [x[0][:20] + '...' if len(x[0]) > 20 else x[0] for x in sorted_values])
            plt.title('Top Values for Canada\'s Future')
            plt.xlabel('Percentage')
        
        # Top concerns
        if 'social_divisions' in analysis_results and 'concern_percentages' in analysis_results['social_divisions']:
            plt.subplot(2, 3, 6)
            concerns = analysis_results['social_divisions']['concern_percentages']
            sorted_concerns = sorted(concerns.items(), key=lambda x: x[1], reverse=True)[:6]
            plt.barh(range(len(sorted_concerns)), [x[1] for x in sorted_concerns])
            plt.yticks(range(len(sorted_concerns)), [x[0][:20] + '...' if len(x[0]) > 20 else x[0] for x in sorted_concerns])
            plt.title('Top Division Concerns')
            plt.xlabel('Percentage')
        
        plt.tight_layout()
        
        # Save as base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations['comprehensive_overview'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        # 2. Theme Analysis Chart
        if 'themes' in analysis_results and 'theme_scores' in analysis_results['themes']:
            plt.figure(figsize=(12, 8))
            themes = analysis_results['themes']['theme_scores']
            sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:12]
            
            plt.barh(range(len(sorted_themes)), [x[1] for x in sorted_themes], color='skyblue')
            plt.yticks(range(len(sorted_themes)), [x[0].replace('_', ' ').title() for x in sorted_themes])
            plt.title('Theme Frequency Analysis')
            plt.xlabel('Mentions')
            
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            visualizations['theme_analysis'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
    except Exception as e:
        print(f"   ⚠️ Error generating visualizations: {e}")
    
    return visualizations

def generate_executive_dashboard(analysis_results):
    """Generate executive dashboard report."""
    dashboard = []
    
    # Header
    dashboard.append("=" * 80)
    dashboard.append("CANADIAN IDENTITY SURVEY - COMPREHENSIVE ANALYSIS DASHBOARD")
    dashboard.append("=" * 80)
    dashboard.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    dashboard.append(f"Data Source: Canadian Identity Survey responses")
    dashboard.append("")
    
    # Key Metrics
    metadata = analysis_results.get('metadata', {})
    dashboard.append("📊 KEY METRICS")
    dashboard.append("-" * 15)
    dashboard.append(f"Total Responses: {metadata.get('total_responses', 0)}")
    
    if 'response_completeness' in metadata:
        completeness = metadata['response_completeness']
        dashboard.append(f"Average Response Completeness: {completeness.get('average_completeness', 0):.1%}")
    
    if 'distribution_types' in metadata:
        methods = metadata['distribution_types']
        dashboard.append(f"Data Collection: {', '.join(f'{k}({v})' for k, v in methods.items())}")
    
    dashboard.append("")
    
    # Identity Connection Analysis
    identity = analysis_results.get('identity_connection', {})
    dashboard.append("🇨🇦 CANADIAN IDENTITY CONNECTION")
    dashboard.append("-" * 35)
    
    if 'average_connection_score' in identity:
        score = identity['average_connection_score']
        strength = identity.get('connection_strength', 'Unknown')
        dashboard.append(f"Overall Connection Strength: {strength} ({score:.2f}/5.0)")
    
    if 'connection_levels' in identity:
        levels = identity['connection_levels']
        total = sum(levels.values())
        very_connected = levels.get('Very connected', 0)
        somewhat_connected = levels.get('Somewhat connected', 0)
        connected_pct = ((very_connected + somewhat_connected) / total) * 100 if total > 0 else 0
        dashboard.append(f"Strong Connection Rate: {connected_pct:.1f}% feel very/somewhat connected")
    
    dashboard.append("")
    
    # Demographics Overview
    demo = analysis_results.get('demographic_patterns', {})
    dashboard.append("👥 DEMOGRAPHIC PROFILE")
    dashboard.append("-" * 22)
    
    if demo.get('provinces'):
        top_provinces = sorted(demo['provinces'].items(), key=lambda x: x[1], reverse=True)[:3]
        dashboard.append(f"Top Provinces: {', '.join([f'{p}({c})' for p, c in top_provinces])}")
    
    if demo.get('generations'):
        top_gens = sorted(demo['generations'].items(), key=lambda x: x[1], reverse=True)[:3]
        dashboard.append(f"Generational Mix: {', '.join([f'{g}({c})' for g, c in top_gens])}")
    
    if demo.get('urban_rural'):
        urban_rural = demo['urban_rural']
        total = sum(urban_rural.values())
        urban_count = urban_rural.get('Urban (big city or downtown area)', 0)
        rural_count = urban_rural.get('Rural (small town, village, or countryside)', 0)
        urban_pct = (urban_count / total) * 100 if total > 0 else 0
        rural_pct = (rural_count / total) * 100 if total > 0 else 0
        dashboard.append(f"Settlement Pattern: {urban_pct:.1f}% Urban, {rural_pct:.1f}% Rural")
    
    dashboard.append("")
    
    # Values Priorities
    values = analysis_results.get('values_priorities', {})
    dashboard.append("⭐ TOP VALUES FOR CANADA'S FUTURE")
    dashboard.append("-" * 35)
    
    if values.get('top_values'):
        for i, (value, count) in enumerate(values['top_values'], 1):
            pct = values.get('values_percentages', {}).get(value, 0)
            dashboard.append(f"  {i}. {value}: {pct:.1f}% ({count} responses)")
    
    dashboard.append("")
    
    # Social Divisions Analysis
    divisions = analysis_results.get('social_divisions', {})
    dashboard.append("⚠️  MAJOR CONCERNS & DIVISIONS")
    dashboard.append("-" * 30)
    
    if divisions.get('top_concerns'):
        for i, (concern, count) in enumerate(divisions['top_concerns'], 1):
            pct = divisions.get('concern_percentages', {}).get(concern, 0)
            status = "🔴 CRITICAL" if pct > 40 else "🟠 MODERATE" if pct > 25 else "🟡 MANAGEABLE"
            dashboard.append(f"  {status}: {concern} ({pct:.1f}%)")
    
    dashboard.append("")
    
    # Connection Building Strategies
    connections = analysis_results.get('connection_building', {})
    dashboard.append("🤝 PREFERRED CONNECTION STRATEGIES")
    dashboard.append("-" * 35)
    
    if connections.get('top_strategies'):
        for i, (strategy, count) in enumerate(connections['top_strategies'], 1):
            pct = connections.get('strategy_percentages', {}).get(strategy, 0)
            dashboard.append(f"  {i}. {strategy}: {pct:.1f}% ({count} responses)")
    
    dashboard.append("")
    
    # Future Outlook
    future = analysis_results.get('future_aspirations', {})
    dashboard.append("🔮 FUTURE OUTLOOK")
    dashboard.append("-" * 17)
    
    if future.get('hopes', {}).get('theme_counts'):
        hope_themes = sorted(future['hopes']['theme_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
        dashboard.append(f"Top Hope Themes: {', '.join([f'{t}({c})' for t, c in hope_themes])}")
    
    if future.get('fears', {}).get('theme_counts'):
        fear_themes = sorted(future['fears']['theme_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
        dashboard.append(f"Top Fear Themes: {', '.join([f'{t}({c})' for t, c in fear_themes])}")
    
    dashboard.append("")
    
    # Top Themes Overall
    themes = analysis_results.get('themes', {})
    dashboard.append("🎯 DOMINANT THEMES")
    dashboard.append("-" * 18)
    
    if themes.get('theme_scores'):
        sorted_themes = sorted(themes['theme_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
        for theme, score in sorted_themes:
            dashboard.append(f"  • {theme.replace('_', ' ').title()}: {score} mentions")
    
    dashboard.append("")
    
    # Critical Insights
    dashboard.append("💡 CRITICAL INSIGHTS")
    dashboard.append("-" * 20)
    
    # Generate insights based on the analysis
    insights = generate_critical_insights(analysis_results)
    for insight in insights:
        dashboard.append(f"  • {insight}")
    
    dashboard.append("")
    
    # Strategic Recommendations
    dashboard.append("🎯 STRATEGIC RECOMMENDATIONS")
    dashboard.append("-" * 30)
    
    recommendations = generate_strategic_recommendations(analysis_results)
    for i, rec in enumerate(recommendations, 1):
        dashboard.append(f"  {i}. {rec}")
    
    dashboard.append("")
    dashboard.append("=" * 80)
    
    return "\n".join(dashboard)

def generate_critical_insights(analysis_results):
    """Generate critical insights from the analysis."""
    insights = []
    
    # Identity connection insights
    identity = analysis_results.get('identity_connection', {})
    if 'average_connection_score' in identity:
        score = identity['average_connection_score']
        if score >= 4.0:
            insights.append("Strong national identity foundation - majority feel connected to Canada")
        elif score < 3.0:
            insights.append("CONCERN: Weak national identity connection requires immediate attention")
        else:
            insights.append("Moderate identity connection with room for strengthening")
    
    # Values insights
    values = analysis_results.get('values_priorities', {})
    if values.get('top_values'):
        top_value = values['top_values'][0][0]
        insights.append(f"Clear value consensus around '{top_value}' as top priority")
    
    # Division insights
    divisions = analysis_results.get('social_divisions', {})
    if divisions.get('top_concerns'):
        top_concern = divisions['top_concerns'][0][0]
        pct = divisions.get('concern_percentages', {}).get(top_concern, 0)
        if pct > 40:
            insights.append(f"CRITICAL: {top_concern} is a major divisive issue ({pct:.1f}% concerned)")
        else:
            insights.append(f"Primary concern is {top_concern} but at manageable levels")
    
    # Demographic insights
    demo = analysis_results.get('demographic_patterns', {})
    if demo.get('diversity_metrics'):
        diversity = demo['diversity_metrics']
        if 'provinces_diversity' in diversity and diversity['provinces_diversity'] < 1.5:
            insights.append("Geographic concentration may limit national representation")
        if 'generations_balance' in diversity and diversity['generations_balance'] > 0.8:
            insights.append("Good generational representation across age groups")
    
    # Future outlook insights
    future = analysis_results.get('future_aspirations', {})
    if future.get('hopes') and future.get('fears'):
        hope_count = future['hopes'].get('total_responses', 0)
        fear_count = future['fears'].get('total_responses', 0)
        if hope_count > fear_count * 1.5:
            insights.append("Optimistic future outlook - hopes significantly outweigh fears")
        elif fear_count > hope_count:
            insights.append("CONCERN: Future fears outweigh hopes - pessimistic outlook")
    
    return insights

def generate_strategic_recommendations(analysis_results):
    """Generate strategic recommendations based on analysis."""
    recommendations = []
    
    # Identity-based recommendations
    identity = analysis_results.get('identity_connection', {})
    if 'average_connection_score' in identity and identity['average_connection_score'] < 3.5:
        recommendations.append("PRIORITY: Launch national identity strengthening initiatives")
    
    # Values-based recommendations
    values = analysis_results.get('values_priorities', {})
    if values.get('top_values'):
        top_value = values['top_values'][0][0]
        recommendations.append(f"Align national policies with citizen priority: {top_value}")
    
    # Division-based recommendations
    divisions = analysis_results.get('social_divisions', {})
    if divisions.get('top_concerns'):
        top_concern = divisions['top_concerns'][0][0]
        pct = divisions.get('concern_percentages', {}).get(top_concern, 0)
        if pct > 35:
            recommendations.append(f"URGENT: Address {top_concern.lower()} through targeted programs")
    
    # Connection-building recommendations
    connections = analysis_results.get('connection_building', {})
    if connections.get('top_strategies'):
        top_strategy = connections['top_strategies'][0][0]
        recommendations.append(f"Implement {top_strategy.lower()} programs nationwide")
    
    # Future-focused recommendations
    future = analysis_results.get('future_aspirations', {})
    if future.get('hopes', {}).get('theme_counts'):
        hope_themes = sorted(future['hopes']['theme_counts'].items(), key=lambda x: x[1], reverse=True)[:2]
        for theme, count in hope_themes:
            recommendations.append(f"Develop initiatives focused on {theme} to align with citizen aspirations")
    
    # Ensure we have at least some recommendations
    if not recommendations:
        recommendations = [
            "Strengthen national unity through civic engagement programs",
            "Address regional disparities through targeted development",
            "Enhance democratic participation and civic education",
            "Foster inclusive communities that welcome diversity",
            "Build future-ready policies based on citizen priorities"
        ]
    
    return recommendations[:6]  # Return top 6 recommendations

def run_comprehensive_analysis(data_dir, output_dir=None):
    """Run complete comprehensive analysis."""
    print("🚀 Starting Canadian Identity Survey Comprehensive Analysis")
    print("=" * 60)
    
    # Set up output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = Path("canadian_identity_analysis_output")
        output_path.mkdir(exist_ok=True)
    
    # Load and validate data
    print("📁 Loading data...")
    datasets = load_and_validate_data(data_dir)
    responses = extract_survey_responses(datasets)
    print(f"✓ Extracted {len(responses)} structured responses")
    
    # Run all analyses
    print("\n🔍 Running analyses...")
    
    print("  • Analyzing response metadata...")
    metadata_analysis = analyze_response_metadata(responses)
    
    print("  • Analyzing demographic patterns...")
    demographic_analysis = analyze_demographic_patterns(responses)
    
    print("  • Analyzing identity connection...")
    identity_analysis = analyze_identity_connection(responses)
    
    print("  • Analyzing values priorities...")
    values_analysis = analyze_values_priorities(responses)
    
    print("  • Analyzing social divisions...")
    divisions_analysis = analyze_social_divisions(responses)
    
    print("  • Analyzing connection building...")
    connections_analysis = analyze_connection_building(responses)
    
    print("  • Analyzing future aspirations...")
    future_analysis = analyze_future_aspirations(responses)
    
    print("  • Analyzing themes...")
    theme_analysis = comprehensive_theme_analysis(responses)
    
    print("  • Extracting actionable insights...")
    insights_analysis = extract_actionable_insights(responses)
    
    # Combine all results
    complete_results = {
        'metadata': metadata_analysis,
        'demographic_patterns': demographic_analysis,
        'identity_connection': identity_analysis,
        'values_priorities': values_analysis,
        'social_divisions': divisions_analysis,
        'connection_building': connections_analysis,
        'future_aspirations': future_analysis,
        'themes': theme_analysis,
        'actionable_insights': insights_analysis,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    print("  • Generating visualizations...")
    visualizations = generate_visualizations(complete_results)
    
    print("  • Creating executive dashboard...")
    dashboard = generate_executive_dashboard(complete_results)
    
    # Save all outputs
    print(f"\n💾 Saving results to {output_path}/")
    
    # Save complete analysis
    with open(output_path / "comprehensive_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)
    print("  ✓ Saved comprehensive_analysis.json")
    
    # Save dashboard
    with open(output_path / "executive_dashboard.txt", 'w', encoding='utf-8') as f:
        f.write(dashboard)
    print("  ✓ Saved executive_dashboard.txt")
    
    # Save visualizations
    if visualizations:
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for viz_name, viz_data in visualizations.items():
            with open(viz_dir / f"{viz_name}.png", 'wb') as f:
                f.write(base64.b64decode(viz_data))
        print(f"  ✓ Saved {len(visualizations)} visualizations")
    
    # Save individual analysis components
    components_dir = output_path / "detailed_analysis"
    components_dir.mkdir(exist_ok=True)
    
    for component_name, component_data in complete_results.items():
        if component_name not in ['metadata', 'analysis_timestamp']:
            with open(components_dir / f"{component_name}.json", 'w', encoding='utf-8') as f:
                json.dump(component_data, f, indent=2, ensure_ascii=False)
    print("  ✓ Saved detailed analysis components")
    
    # Display dashboard
    print("\n" + "="*80)
    print("EXECUTIVE DASHBOARD")
    print("="*80)
    print(dashboard)
    
    print(f"\n🎉 Analysis complete! Results saved to: {output_path.absolute()}")
    
    return complete_results

def main():
    parser = argparse.ArgumentParser(description="Canadian Identity Survey Comprehensive Analysis")
    parser.add_argument("--data-dir", required=True,
                      help="Directory containing Canadian Identity Survey CSV files")
    parser.add_argument("--output-dir", default="canadian_identity_analysis_output",
                      help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    try:
        run_comprehensive_analysis(args.data_dir, args.output_dir)
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
  