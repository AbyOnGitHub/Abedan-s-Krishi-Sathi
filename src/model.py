import pandas as pd
import re

class AIEligibilityScoringEngine:
    """
    AI Eligibility Scoring Engine applying a rule-based scoring model 
    based on farmer's socio-economic and farming attributes.
    """
    def __init__(self, schemes_data_path):
        self.schemes_df = pd.read_csv(schemes_data_path)
        
    def categorize_scheme(self, name, benefits):
        """Categorize scheme to provide diverse recommendations."""
        text = str(name).lower() + " " + str(benefits).lower()
        if any(k in text for k in ['bima', 'insurance']):
            return 'Crop Insurance'
        elif any(k in text for k in ['sinchai', 'irrigation', 'pump', 'drip', 'sprinkler', 'pipeline', 'water']):
            return 'Irrigation Support'
        elif any(k in text for k in ['tractor', 'equipment', 'mechanization', 'machine', 'harvester']):
            return 'Farm Equipment & Mechanization'
        elif any(k in text for k in ['seed', 'orchard', 'nursery', 'flower', 'spice', 'mushroom', 'beekeeping', 'vermicompost', 'tissue culture', 'oilseed', 'farming']):
            return 'Crop & Infrastructure'
        elif any(k in text for k in ['soil health', 'plant health', 'natural', 'ayush', 'testing']):
            return 'Health & Soil Management'
        else:
            return 'Direct Financial Support / General'

    def score_scheme(self, farmer, scheme):
        score = 0
        relevance_score = 0
        priority_score = 0
        reasons = [] # Explanations for why eligible/ineligible or points given
        
        # 1. Base Eligibility
        scheme_state = scheme['state']
        if scheme_state != 'Nationwide' and scheme_state.lower() != farmer['state'].lower():
            reasons.append(f"Ineligible: Scheme is specific to {scheme_state}, but farmer is in {farmer['state']}.")
            return 0, reasons, False
            
        relevance_score += 30
        reasons.append("State Match: Farmer state aligns with scheme applicability (+30).")
        
        eligibility_text = str(scheme['eligibility']).lower()
        benefits_text = str(scheme['benefits']).lower()
        
        # 2. Extract and check criteria
        
        # Land Size Criteria
        if "small and marginal" in eligibility_text or "up to 2 hectares" in eligibility_text:
            if farmer['land_size'] <= 2.0:
                relevance_score += 20
                reasons.append(f"Land Target Match: Farmer's land size ({farmer['land_size']} ha) qualifies as small/marginal (+20).")
            else:
                reasons.append(f"Ineligible: Scheme requires small/marginal land (<= 2 ha), but farmer owns {farmer['land_size']} ha.")
                return 0, reasons, False

        # Landless Criteria
        if "landless" in eligibility_text:
            if farmer['land_size'] == 0:
                relevance_score += 20
                reasons.append("Land Target Match: Farmer is landless, exactly matching target group (+20).")
            else:
                relevance_score -= 10
                reasons.append("Constraint Penalty: Scheme targets landless, but farmer owns land (-10).")
                if relevance_score < 0:
                    return 0, reasons, False

        # Income filter (approximation if scheme mentions low income/pensioners)
        if "income tax" in eligibility_text:
            if farmer.get('annual_income', 0) > 500000: # arbitrary tax threshold
                reasons.append(f"Ineligible: Farmer's income (Rs {farmer.get('annual_income')}) may exceed income tax payer limits.")
                return 0, reasons, False
            elif farmer.get('annual_income') != None:
                reasons.append("Income Criteria: Income is assumed within non-taxable limits.")
                
        # Category/Caste Criteria
        if "sc/st" in eligibility_text or "obc" in eligibility_text:
            if farmer.get('cast_category', '').upper() in ['SC', 'ST', 'OBC']:
                relevance_score += 15
                reasons.append(f"Demographic Priority: Farmer's category ({farmer.get('cast_category')}) receives targeted priority (+15).")
                
        # Gender Criteria
        if "women" in eligibility_text:
            if farmer.get('gender', '').lower() == 'female':
                relevance_score += 15
                reasons.append("Demographic Priority: Farmer identifies as female, matching demographic target (+15).")
            else:
                reasons.append("Constraint Penalty: Scheme explicitly targets women farmers (-10).")
                relevance_score -= 10
                
        # Crop Type Matching
        if "horticulture" in eligibility_text or "orchard" in eligibility_text or "flower" in eligibility_text or "spice" in eligibility_text or "vegetable" in eligibility_text:
            crop = farmer.get('crop_type', '').lower()
            if crop in ['horticulture', 'fruits', 'vegetables', 'spices', 'flowers']:
                relevance_score += 15
                reasons.append(f"Crop Match: Farmer's crop type ({farmer.get('crop_type')}) aligns closely with scheme intent (+15).")
                
        # Irrigation Type
        if "drip" in eligibility_text or "sprinkler" in eligibility_text or "irrigation" in eligibility_text:
            irrigation = farmer.get('irrigation_type', '').lower()
            if irrigation in ['rainfed', 'none', 'traditional']:
                relevance_score += 15
                reasons.append(f"Infrastructure Match: Scheme provides required irrigation upgrade for currently {irrigation} land (+15).")

        # Age Criteria
        age_match = re.search(r'aged (\d+)-(\d+) years', eligibility_text)
        if age_match:
            min_age = int(age_match.group(1))
            max_age = int(age_match.group(2))
            if min_age <= farmer.get('age', 0) <= max_age:
                relevance_score += 20
                reasons.append(f"Age Match: Age ({farmer.get('age')}) is within required {min_age}-{max_age} range (+20).")
            else:
                reasons.append(f"Ineligible: Age ({farmer.get('age')}) is outside required {min_age}-{max_age} range.")
                return 0, reasons, False
                
        # 3. Priority Scoring
        if "rs" in benefits_text or "financial" in benefits_text or "subsidy" in benefits_text:
            priority_score += 10
            reasons.append("Benefit Value: Scheme provides direct financial aid or robust subsidy (+10).")
            
        if "pension" in benefits_text:
            priority_score += 15
            reasons.append("Benefit Value: Scheme offers long-term pension security (+15).")
            
        if "insurance" in benefits_text or "risk" in benefits_text:
            priority_score += 15
            reasons.append("Benefit Value: Scheme functions as crucial crop/life insurance mitigation (+15).")
            
        if "10,000" in benefits_text or "10000" in benefits_text:
            priority_score += 10
            reasons.append("Benefit Value: Noted as a high-value financial transfer (+10).")
            
        total_score = relevance_score + priority_score
        return total_score, reasons, True
        
    def get_recommendations(self, farmer, top_n=5, diversity_mode=True):
        """
        Returns a diversified set of scheme recommendations based on category.
        Includes a log of reasons explaining why each scheme was chosen.
        """
        scored_schemes = []
        for index, row in self.schemes_df.iterrows():
            score, reasons, is_el = self.score_scheme(farmer, row)
            cat = self.categorize_scheme(row['scheme_name'], row['benefits'])
            
            scored_schemes.append({
                'id': row['id'],
                'scheme_name': row['scheme_name'],
                'state': row['state'],
                'category': cat,
                'score': score,
                'reasons': reasons,
                'is_eligible': is_el,
                'benefits': row['benefits']
            })
                
        eligible_schemes = [s for s in scored_schemes if s['is_eligible']]
        eligible_schemes.sort(key=lambda x: x['score'], reverse=True)
        
        if not diversity_mode:
            return eligible_schemes[:top_n]
            
        # Group top schemes by category for diversified recommendations
        best_by_category = {}
        for s in eligible_schemes:
            if s['category'] not in best_by_category:
                best_by_category[s['category']] = []
            # Take top 2 from each category max
            if len(best_by_category[s['category']]) < 2:
                best_by_category[s['category']].append(s)
                
        # Flatten and sort the diversified recommendations by score
        final_recs = []
        for cat, schemes in best_by_category.items():
            final_recs.extend(schemes)
            
        final_recs.sort(key=lambda x: x['score'], reverse=True)
        return final_recs[:top_n]

if __name__ == "__main__":
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'farmer_schemes.csv')
    engine = AIEligibilityScoringEngine(data_path)
    
    # Simulating a more robust farmer profile based on new requirements
    test_farmer = {
        'state': 'Uttar Pradesh',
        'district': 'Lucknow',
        'village': 'Sample Village',
        'land_size': 1.5,
        'crop_type': 'Vegetables',
        'irrigation_type': 'Rainfed', # Traditional/Rainfed
        'annual_income': 45000,
        'cast_category': 'SC',
        'age': 35,
        'gender': 'Male',
        'created_at': '2023-11-01',
        'updated_at': '2023-11-01'
    }
    
    print("\n" + "="*80)
    print(f"Generating recommendations for Farmer:\nState: {test_farmer['state']}, Land: {test_farmer['land_size']} ha, Crop: {test_farmer['crop_type']}, Category: {test_farmer['cast_category']}")
    print("="*80)
    
    recommendations = engine.get_recommendations(test_farmer, top_n=5, diversity_mode=True)
    
    for rank, scheme in enumerate(recommendations, 1):
        print(f"\n[{rank}] {scheme['scheme_name']} ({scheme['state']})")
        print(f"    Category: {scheme['category']} | Score: {scheme['score']}")
        print(f"    Benefits: {scheme['benefits'][:120]}...")
        print("    Justifications for Eligibility Profile:")
        for reason in scheme['reasons']:
            print(f"      - {reason}")
