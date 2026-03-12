import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
import warnings

# Suppress lightgbm warnings for cleaner output
warnings.filterwarnings('ignore')

class AIEligibilityScoringEngine:
    """
    Advanced AI Eligibility Scoring Engine integrating rule-based heuristics, 
    TF-IDF semantic similarity, and LightGBM for robust ranking models.
    """
    def __init__(self, schemes_data_path):
        self.schemes_df = pd.read_csv(schemes_data_path)
        self.schemes_df['eligibility'] = self.schemes_df['eligibility'].astype(str).fillna('')
        self.schemes_df['benefits'] = self.schemes_df['benefits'].astype(str).fillna('')
        
        # 1. Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        
        # Combine scheme text for corpus (Name + Eligibility + Benefits)
        corpus = self.schemes_df['scheme_name'] + " " + self.schemes_df['eligibility'] + " " + self.schemes_df['benefits']
        self.scheme_tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        # 2. Train LightGBM Ranker Model
        self._train_lgbm_model()
        
    def _train_lgbm_model(self):
        """
        Creates synthetic training data from our rule-based logic to train a LightGBM Ranker.
        In a real-world scenario, this would use historical application success rates.
        """
        # Create a synthetic dataset of 'simulated farmers'
        synthetic_farmers = [
            {'state': 'Uttar Pradesh', 'land_size': 1.5, 'age': 35, 'annual_income': 40000, 'crop_type': 'Vegetables', 'irrigation_type': 'Rainfed', 'cast_category': 'SC', 'gender': 'Male'},
            {'state': 'Maharashtra', 'land_size': 3.0, 'age': 50, 'annual_income': 150000, 'crop_type': 'Cotton', 'irrigation_type': 'Tube well', 'cast_category': 'General', 'gender': 'Female'},
            {'state': 'Nationwide', 'land_size': 0.5, 'age': 28, 'annual_income': 20000, 'crop_type': 'Spices', 'irrigation_type': 'None', 'cast_category': 'ST', 'gender': 'Male'},
            {'state': 'Bihar', 'land_size': 0.0, 'age': 45, 'annual_income': 10000, 'crop_type': 'None', 'irrigation_type': 'None', 'cast_category': 'OBC', 'gender': 'Male'},
        ]
        
        train_features = []
        train_labels = [] # We use the rule-based Relevance + Priority score as the "ground truth" label to train LightGBM for ranking
        train_group = []  # Number of schemes per query (required for LGBMRanker)
        
        for farmer in synthetic_farmers:
            farmer_profile_text = f"{farmer['state']} farmer {farmer['land_size']} hectares {farmer['crop_type']} {farmer['irrigation_type']} {farmer['cast_category']} {farmer['gender']}"
            farmer_vector = self.tfidf_vectorizer.transform([farmer_profile_text])
            
            # Calculate Cosine Similarity against all schemes natively
            cosine_sims = cosine_similarity(farmer_vector, self.scheme_tfidf_matrix).flatten()
            
            group_count = 0
            for idx, row in self.schemes_df.iterrows():
                # 1. Feature: TF-IDF Similarity
                tf_idf_score = cosine_sims[idx]
                
                # 2. Extract Rule-based scores for training features
                rule_based_total, _, is_el = self._calculate_rule_based_scores(farmer, row)
                
                # We skip strictly ineligible to keep training clean
                if not is_el:
                    continue
                    
                # The label LightGBM uses to rank is our historical/rule-based score
                target_score = rule_based_total 
                
                # Features for LightGBM
                features = [
                    tf_idf_score,
                    farmer['land_size'],
                    farmer['age'],
                    farmer['annual_income'],
                    1 if str(row['state']) == 'Nationwide' or str(row['state']).lower() == farmer['state'].lower() else 0, # State Match Indicator
                ]
                
                train_features.append(features)
                train_labels.append(target_score)
                group_count += 1
                
            if group_count > 0:
                train_group.append(group_count)
                
        # Convert continuous scores into discrete relevance labels (0 to 4) for LambdaRank
        # We process it per group to rank the schemes relative to each other for that specific farmer
        discrete_labels = []
        start_idx = 0
        for group_size in train_group:
            group_scores = train_labels[start_idx:start_idx+group_size]
            # Create discrete bins (0 to 4) based on quantiles within the group,
            # or just simple ranking if all scores are identical
            if len(set(group_scores)) > 1:
                # Rank data and map to 5 buckets (0, 1, 2, 3, 4)
                ranks = pd.qcut(group_scores, q=5, labels=False, duplicates='drop')
                discrete_labels.extend(ranks)
            else:
                discrete_labels.extend([0] * group_size)
            start_idx += group_size
            
        train_labels = discrete_labels

        X = np.array(train_features)
        y = np.array(train_labels)
        # Add slight integer variance to ensure LightGBM can split
        np.random.seed(42)
        # Randomly bump up some labels by 1 to create ranking splits (0, 1, 2, 3, 4, 5)
        # Ensure it remains integer format for LGBMRanker
        y = y + np.random.randint(0, 2, size=len(y))
        y = y.astype(int)

        # Fit the LightGBM Ranker
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[i for i in range(max(y) + 1)], # Ensure label mappings match our discrete buckets after variance addition
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        )
        
        if len(X) > 0:
            self.ranker.fit(X, y, group=train_group)
        else:
            print("Warning: Insufficient data to train LightGBM.")
            
    def categorize_scheme(self, name, benefits):
        text = str(name).lower() + " " + str(benefits).lower()
        if any(k in text for k in ['bima', 'insurance']): return 'Crop Insurance'
        elif any(k in text for k in ['sinchai', 'irrigation', 'pump', 'drip', 'sprinkler']): return 'Irrigation Support'
        elif any(k in text for k in ['tractor', 'equipment', 'mechanization', 'machine']): return 'Farm Equipment'
        elif any(k in text for k in ['seed', 'orchard', 'nursery', 'flower', 'spice']): return 'Crop & Infrastructure'
        elif any(k in text for k in ['soil health', 'plant health', 'natural', 'ayush']): return 'Health & Soil Management'
        else: return 'Direct Financial Support'

    def _calculate_rule_based_scores(self, farmer, scheme):
        """Internal method retaining the hard rules to catch strict bounds and create priority baselines."""
        relevance_score = 0
        priority_score = 0
        reasons = []
        
        scheme_state = scheme['state']
        if scheme_state != 'Nationwide' and scheme_state.lower() != farmer['state'].lower():
            reasons.append(f"Ineligible: Scheme is specific to {scheme_state}.")
            return 0, reasons, False
            
        reasons.append("State Match: Farmer state aligns (+30).")
        relevance_score += 30
        
        eligibility_text = str(scheme['eligibility']).lower()
        benefits_text = str(scheme['benefits']).lower()
        
        if "small and marginal" in eligibility_text or "up to 2 hectares" in eligibility_text:
            if farmer['land_size'] <= 2.0: relevance_score += 20
            else: return 0, ["Ineligible: Scheme requires <= 2 ha land."], False
                
        if "landless" in eligibility_text:
            if farmer['land_size'] == 0: relevance_score += 20
            else: relevance_score -= 10
                
        if "rs" in benefits_text or "financial" in benefits_text: priority_score += 10
        if "pension" in benefits_text: priority_score += 15
        if "insurance" in benefits_text: priority_score += 15
        
        return relevance_score + priority_score, reasons, True
        
    def get_recommendations(self, farmer, top_n=5):
        """
        Uses LightGBM and TF-IDF to predict and rank the best schemes.
        """
        # Create farmer TF-IDF vector
        farmer_profile_text = f"{farmer.get('state','')} farmer {farmer.get('land_size',0)} hectares {farmer.get('crop_type','')} {farmer.get('irrigation_type','')} {farmer.get('cast_category','')} {farmer.get('gender','')}"
        farmer_vector = self.tfidf_vectorizer.transform([farmer_profile_text])
        cosine_sims = cosine_similarity(farmer_vector, self.scheme_tfidf_matrix).flatten()
        
        scored_schemes = []
        inference_features = []
        valid_indices = []
        
        # First Pass: Run strict base exclusions and build inference array
        for idx, row in self.schemes_df.iterrows():
            rule_score, reasons, is_el = self._calculate_rule_based_scores(farmer, row)
            
            if not is_el:
                continue # Skip strictly disqualified
                
            state_match = 1 if str(row['state']) == 'Nationwide' or str(row['state']).lower() == farmer['state'].lower() else 0
            tf_idf_score = cosine_sims[idx]
            
            features = [
                tf_idf_score,
                farmer.get('land_size', 0),
                farmer.get('age', 30),
                farmer.get('annual_income', 50000),
                state_match
            ]
            
            inference_features.append(features)
            # Retain rule_score to use if ML score fails to differentiate
            valid_indices.append((idx, row, reasons, tf_idf_score, rule_score))
            
        # Second Pass: Predict Rankings using LightGBM
        if len(inference_features) > 0:
            predictions = self.ranker.predict(np.array(inference_features))
            
            # Map predictions back to schemes
            for i, (idx, row, reasons, tf_idf_score, rule_score) in enumerate(valid_indices):
                lgbm_score = float(predictions[i])
                
                reasons.append(f"TF-IDF Semantic Similarity Score: {tf_idf_score:.3f}")
                
                # Combine LGBM and Rule output for robust ranking since dataset is entirely synthetic
                final_score = (lgbm_score * 100) + rule_score + (tf_idf_score * 50)
                reasons.append(f"LightGBM Prediction Score: {lgbm_score:.3f} | Rule Baseline: {rule_score:.1f}")
                
                scored_schemes.append({
                    'id': row['id'],
                    'scheme_name': row['scheme_name'],
                    'state': row['state'],
                    'category': self.categorize_scheme(row['scheme_name'], row['benefits']),
                    'final_ml_score': final_score, 
                    'reasons': reasons,
                    'benefits': row['benefits'][:100] + "..."
                })
                
        # Sort globally relying fully on the composite ML model predictions
        scored_schemes.sort(key=lambda x: x['final_ml_score'], reverse=True)
        return scored_schemes[:top_n]

if __name__ == "__main__":
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'farmer_schemes.csv')
    
    # Needs to process dataset first to build TF-IDF matrix and Train LightGBM
    print("Training LightGBM model and compiling TF-IDF embeddings...")
    engine = AIEligibilityScoringEngine(data_path)
    print("Models successfully deployed into scoring engine.\n")
    
    test_farmer = {
        'state': 'Uttar Pradesh',
        'district': 'Lucknow',
        'land_size': 1.5,
        'crop_type': 'Vegetables',
        'irrigation_type': 'Rainfed',
        'annual_income': 45000,
        'cast_category': 'SC',
        'age': 35,
        'gender': 'Male'
    }
    
    print("="*80)
    print(f"Generating Machine Learning Recommendations for Farmer:\nState: {test_farmer['state']}, Land: {test_farmer['land_size']} ha, Crop: {test_farmer['crop_type']}")
    print("="*80)
    
    recommendations = engine.get_recommendations(test_farmer, top_n=5)
    
    for rank, scheme in enumerate(recommendations, 1):
        print(f"\n[{rank}] {scheme['scheme_name']} ({scheme['state']})")
        print(f"    Category: {scheme['category']}")
        print(f"    ML Final Rank Score: {scheme['final_ml_score']:.4f}")
        for reason in scheme['reasons']:
            print(f"      - {reason}")
