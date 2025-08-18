import csv
import random
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import uuid

# Initialize Faker
fake = Faker()

# Platform configuration - matching your system
USER_PROVIDER_ID = "PROV-A20E5610"  # Your provider ID
num_records = 3000  # Total number of bids
num_providers = 30  # Number of different providers
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
output_file = "enhanced_bidding_data.csv"

# Enhanced field names for rich analytics
fieldnames = [
    'bid_id', 'price', 'provider_id', 'win_loss', 'quality_score', 'delivery_time', 
    'date', 'complexity', 'project_category', 'winning_provider_id', 'winning_price', 
    'num_bids', 'client_id', 'client_industry', 'client_size', 'project_value',
    'client_relationship_score', 'geographic_region', 'competition_intensity',
    'market_conditions', 'seasonal_factor', 'urgency_level', 'technical_difficulty',
    'past_client_experience', 'proposal_quality', 'team_experience', 'innovation_score',
    'risk_level', 'payment_terms', 'contract_duration', 'client_budget_range',
    'incumbent_advantage', 'strategic_importance', 'reference_strength',
    # Industry-specific metadata
    'material_type', 'material_cost', 'material_quantity', 'unit_size', 'total_units',
    'factory_location', 'production_capacity', 'shipping_cost', 'shipping_method',
    'lead_time_days', 'warehousing_cost', 'customs_duties', 'quality_certification',
    'sustainability_rating', 'supplier_tier'
]

def generate_enhanced_bidding_data(num_records):
    """Generate comprehensive B2B proposal bidding data"""
    data = []
    
    # Create provider ecosystem
    provider_profiles = create_provider_profiles()
    client_profiles = create_client_profiles()
    
    # Create realistic project/auction groups
    num_projects = int(num_records / 5.5)  # Average 5.5 bids per project
    
    for project_idx in range(num_projects):
        project_data = generate_project(project_idx, provider_profiles, client_profiles)
        data.extend(project_data)
    
    return data

def create_provider_profiles():
    """Create detailed provider profiles with different strengths"""
    providers = {}
    
    # Provider categories with different characteristics
    provider_types = {
        'enterprise': {'reputation': (0.8, 1.2), 'price_strategy': (1.1, 1.3), 'quality': (75, 95)},
        'specialist': {'reputation': (0.7, 1.1), 'price_strategy': (1.0, 1.2), 'quality': (70, 90)},
        'cost_leader': {'reputation': (0.6, 0.9), 'price_strategy': (0.7, 0.9), 'quality': (50, 75)},
        'boutique': {'reputation': (0.8, 1.1), 'price_strategy': (1.2, 1.5), 'quality': (80, 98)},
        'emerging': {'reputation': (0.5, 0.8), 'price_strategy': (0.8, 1.1), 'quality': (55, 80)}
    }
    
    # Generate provider IDs
    provider_ids = [USER_PROVIDER_ID]  # Ensure user provider is included
    provider_ids.extend([f"PROV-{uuid.uuid4().hex[:8].upper()}" for _ in range(num_providers - 1)])
    
    for i, provider_id in enumerate(provider_ids):
        if provider_id == USER_PROVIDER_ID:
            # User provider - balanced profile with room for optimization
            provider_type = 'specialist'
        else:
            provider_type = random.choice(list(provider_types.keys()))
        
        profile = provider_types[provider_type]
        
        providers[provider_id] = {
            'type': provider_type,
            'reputation': random.uniform(*profile['reputation']),
            'price_strategy': random.uniform(*profile['price_strategy']),
            'base_quality': random.uniform(*profile['quality']),
            'specializations': random.sample([
                'Manufacturing', 'Technology', 'Healthcare', 'Finance', 'Energy',
                'Retail', 'Logistics', 'Aerospace', 'Automotive', 'Telecommunications'
            ], random.randint(2, 4)),
            'team_size': random.randint(10, 500),
            'years_experience': random.randint(3, 25),
            'geographic_strength': random.choice(['North America', 'Europe', 'Asia-Pacific', 'Global']),
            'innovation_focus': random.uniform(0.3, 1.0),
            'delivery_reliability': random.uniform(0.6, 1.0)
        }
    
    return providers

def create_client_profiles():
    """Create detailed client profiles with preferences and behaviors"""
    clients = {}
    
    # Client industries with different characteristics and specific material needs
    industries = {
        'Manufacturing': {
            'budget_range': (50000, 500000), 
            'price_sensitivity': 0.7, 
            'quality_focus': 0.8,
            'materials': ['Steel', 'Aluminum', 'Plastics', 'Composites', 'Rubber'],
            'typical_sizes': ['Small Parts', 'Medium Components', 'Large Assemblies'],
            'factories': ['China', 'Vietnam', 'Mexico', 'USA', 'Germany']
        },
        'Construction': {
            'budget_range': (100000, 2000000),
            'price_sensitivity': 0.6,
            'quality_focus': 0.75,
            'materials': ['Concrete', 'Steel Beams', 'Lumber', 'Glass', 'Insulation', 'Copper Wiring'],
            'typical_sizes': ['Residential', 'Commercial', 'Infrastructure'],
            'factories': ['Local', 'Regional', 'National']
        },
        'Technology': {
            'budget_range': (75000, 750000), 
            'price_sensitivity': 0.4, 
            'quality_focus': 0.9,
            'materials': ['Silicon Wafers', 'PCB Boards', 'Rare Earth Metals', 'Lithium', 'Copper'],
            'typical_sizes': ['Chips', 'Components', 'Devices', 'Systems'],
            'factories': ['Taiwan', 'South Korea', 'China', 'USA', 'Japan']
        },
        'Healthcare': {
            'budget_range': (100000, 800000), 
            'price_sensitivity': 0.3, 
            'quality_focus': 0.95,
            'materials': ['Medical Grade Plastics', 'Stainless Steel', 'Titanium', 'Silicone', 'Pharmaceuticals'],
            'typical_sizes': ['Consumables', 'Instruments', 'Equipment', 'Implants'],
            'factories': ['USA', 'Germany', 'Switzerland', 'Japan', 'Ireland']
        },
        'Automotive': {
            'budget_range': (200000, 3000000),
            'price_sensitivity': 0.65,
            'quality_focus': 0.85,
            'materials': ['Steel', 'Aluminum', 'Carbon Fiber', 'Rubber', 'Glass', 'Electronics'],
            'typical_sizes': ['Parts', 'Subsystems', 'Complete Vehicles'],
            'factories': ['Mexico', 'USA', 'Germany', 'Japan', 'China', 'South Korea']
        },
        'Finance': {
            'budget_range': (150000, 1000000), 
            'price_sensitivity': 0.5, 
            'quality_focus': 0.85,
            'materials': ['Software Licenses', 'Cloud Infrastructure', 'Security Hardware'],
            'typical_sizes': ['Department', 'Division', 'Enterprise'],
            'factories': ['N/A']
        },
        'Energy': {
            'budget_range': (200000, 2000000), 
            'price_sensitivity': 0.6, 
            'quality_focus': 0.9,
            'materials': ['Solar Panels', 'Wind Turbine Components', 'Batteries', 'Transformers', 'Cables'],
            'typical_sizes': ['Residential', 'Commercial', 'Utility Scale'],
            'factories': ['China', 'USA', 'Germany', 'India', 'Denmark']
        },
        'Retail': {
            'budget_range': (30000, 300000), 
            'price_sensitivity': 0.8, 
            'quality_focus': 0.6,
            'materials': ['Packaging', 'Display Materials', 'RFID Tags', 'Shelving'],
            'typical_sizes': ['Store', 'Regional', 'National'],
            'factories': ['China', 'Vietnam', 'Bangladesh', 'Turkey']
        },
        'Logistics': {
            'budget_range': (40000, 400000), 
            'price_sensitivity': 0.75, 
            'quality_focus': 0.7,
            'materials': ['Pallets', 'Containers', 'Tracking Devices', 'Packaging Materials'],
            'typical_sizes': ['Local', 'Regional', 'International'],
            'factories': ['Various']
        },
        'Aerospace': {
            'budget_range': (500000, 5000000), 
            'price_sensitivity': 0.2, 
            'quality_focus': 0.98,
            'materials': ['Titanium', 'Carbon Composites', 'Aluminum Alloys', 'Specialty Steels', 'Ceramics'],
            'typical_sizes': ['Components', 'Subsystems', 'Complete Aircraft'],
            'factories': ['USA', 'France', 'UK', 'Canada', 'Brazil']
        }
    }
    
    # Client sizes
    client_sizes = ['Startup', 'SME', 'Mid-Market', 'Enterprise', 'Fortune 500']
    
    # Generate client profiles
    for i in range(50):  # 50 different clients
        client_id = f"CLIENT-{uuid.uuid4().hex[:6].upper()}"
        industry = random.choice(list(industries.keys()))
        industry_profile = industries[industry]
        
        clients[client_id] = {
            'industry': industry,
            'size': random.choice(client_sizes),
            'budget_range': industry_profile['budget_range'],
            'price_sensitivity': industry_profile['price_sensitivity'] + random.uniform(-0.2, 0.2),
            'quality_focus': industry_profile['quality_focus'] + random.uniform(-0.1, 0.1),
            'decision_speed': random.uniform(0.3, 1.0),
            'relationship_importance': random.uniform(0.4, 0.9),
            'innovation_appetite': random.uniform(0.2, 0.8),
            'geographic_region': random.choice(['North America', 'Europe', 'Asia-Pacific', 'Latin America']),
            'procurement_sophistication': random.uniform(0.3, 1.0),
            'loyalty_factor': random.uniform(0.2, 0.8)
        }
    
    return clients

def generate_project(project_idx, provider_profiles, client_profiles):
    """Generate a complete project with multiple bids"""
    
    # Select client and project characteristics
    client_id = random.choice(list(client_profiles.keys()))
    client = client_profiles[client_id]
    
    # Generate project details
    project_categories = [
        'Software Development', 'Data Analytics', 'Cloud Migration', 'Digital Transformation',
        'Process Optimization', 'System Integration', 'Cybersecurity', 'AI/ML Implementation',
        'Business Intelligence', 'ERP Implementation', 'Mobile App Development', 'E-commerce Platform',
        'Supply Chain Optimization', 'Quality Management', 'Compliance Automation'
    ]
    
    category = random.choice(project_categories)
    complexity = random.randint(1, 10)
    
    # Project date with seasonal patterns
    days_between = (end_date - start_date).days
    random_day = random.randint(0, days_between)
    project_date = start_date + timedelta(days=random_day)
    
    # Seasonal factors (Q4 budget pressure, Q1 new initiatives)
    month = project_date.month
    if month in [10, 11, 12]:  # Q4 - budget pressure
        seasonal_factor = random.uniform(1.1, 1.3)
        urgency_multiplier = 1.2
    elif month in [1, 2]:  # Q1 - new initiatives
        seasonal_factor = random.uniform(0.9, 1.1)
        urgency_multiplier = 0.8
    else:
        seasonal_factor = random.uniform(0.95, 1.15)
        urgency_multiplier = 1.0
    
    # Project value based on client budget and complexity
    base_value = random.uniform(*client['budget_range'])
    project_value = base_value * (0.5 + complexity/20) * seasonal_factor
    
    # Market conditions (economic factors)
    market_conditions = random.choice(['Recession', 'Recovery', 'Growth', 'Peak'])
    market_multiplier = {'Recession': 0.8, 'Recovery': 0.9, 'Growth': 1.1, 'Peak': 1.2}[market_conditions]
    
    # Determine number of bidders based on project value and market conditions
    if project_value > 500000:
        num_bids = random.randint(4, 8)  # High-value projects attract more bidders
    elif project_value > 100000:
        num_bids = random.randint(3, 6)
    else:
        num_bids = random.randint(2, 5)
    
    # Select bidders - ensure user provider has a chance to bid
    available_providers = list(provider_profiles.keys())
    
    # User provider bids on projects that match their profile with some probability
    user_provider = provider_profiles[USER_PROVIDER_ID]
    specialization_match = category.split()[0] in [spec.split()[0] for spec in user_provider['specializations']]
    bid_probability = 0.7 if specialization_match else 0.3
    
    if random.random() < bid_probability:
        project_bidders = [USER_PROVIDER_ID]
        project_bidders.extend(random.sample([p for p in available_providers if p != USER_PROVIDER_ID], num_bids - 1))
    else:
        project_bidders = random.sample(available_providers, num_bids)
    
    # Generate bids for each provider
    bids = []
    for provider_id in project_bidders:
        bid_data = generate_bid(provider_id, provider_profiles[provider_id], client, client_id, 
                               category, complexity, project_value, seasonal_factor, 
                               market_conditions, project_date, urgency_multiplier)
        bids.append(bid_data)
    
    # Determine winner using sophisticated scoring
    winner_data = determine_winner(bids, client)
    winning_provider = winner_data['provider_id']
    winning_price = winner_data['price']
    
    # Generate industry-specific metadata for this project
    industry_metadata = generate_industry_metadata(
        client['industry'], 
        project_value, 
        complexity, 
        client['geographic_region']
    )
    
    # Create final bid records
    project_bids = []
    for bid in bids:
        provider_id = bid['provider_id']
        win_loss = 'win' if provider_id == winning_provider else 'loss'
        
        # Generate unique bid ID
        bid_id = f"BID-{uuid.uuid4().hex[:10].upper()}"
        
        bid_record = {
            'bid_id': bid_id,
            'price': round(bid['price'], 2),
            'provider_id': provider_id,
            'win_loss': win_loss,
            'quality_score': bid['quality_score'],
            'delivery_time': bid['delivery_time'],
            'date': project_date.strftime("%Y-%m-%d"),
            'complexity': complexity,
            'project_category': category,
            'winning_provider_id': winning_provider,
            'winning_price': round(winning_price, 2),
            'num_bids': num_bids,
            'client_id': client_id,
            'client_industry': client['industry'],
            'client_size': client['size'],
            'project_value': round(project_value, 2),
            'client_relationship_score': bid['relationship_score'],
            'geographic_region': client['geographic_region'],
            'competition_intensity': calculate_competition_intensity(num_bids, project_value),
            'market_conditions': market_conditions,
            'seasonal_factor': round(seasonal_factor, 2),
            'urgency_level': bid['urgency_level'],
            'technical_difficulty': bid['technical_difficulty'],
            'past_client_experience': bid['past_experience'],
            'proposal_quality': bid['proposal_quality'],
            'team_experience': bid['team_experience'],
            'innovation_score': bid['innovation_score'],
            'risk_level': bid['risk_level'],
            'payment_terms': bid['payment_terms'],
            'contract_duration': bid['contract_duration'],
            'client_budget_range': f"${client['budget_range'][0]:,.0f}-${client['budget_range'][1]:,.0f}",
            'incumbent_advantage': bid['incumbent_advantage'],
            'strategic_importance': bid['strategic_importance'],
            'reference_strength': bid['reference_strength'],
            # Add industry-specific metadata
            'material_type': industry_metadata['material_type'],
            'material_cost': industry_metadata['material_cost'],
            'material_quantity': industry_metadata['material_quantity'],
            'unit_size': industry_metadata['unit_size'],
            'total_units': industry_metadata['total_units'],
            'factory_location': industry_metadata['factory_location'],
            'production_capacity': industry_metadata['production_capacity'],
            'shipping_cost': industry_metadata['shipping_cost'],
            'shipping_method': industry_metadata['shipping_method'],
            'lead_time_days': industry_metadata['lead_time_days'],
            'warehousing_cost': industry_metadata['warehousing_cost'],
            'customs_duties': industry_metadata['customs_duties'],
            'quality_certification': industry_metadata['quality_certification'],
            'sustainability_rating': industry_metadata['sustainability_rating'],
            'supplier_tier': industry_metadata['supplier_tier']
        }
        
        project_bids.append(bid_record)
    
    return project_bids

def generate_bid(provider_id, provider_profile, client, client_id, category, complexity, 
                project_value, seasonal_factor, market_conditions, project_date, urgency_multiplier):
    """Generate detailed bid for a specific provider"""
    
    # Base pricing strategy
    base_price = project_value * provider_profile['price_strategy']
    
    # Quality score influenced by provider reputation and client focus
    base_quality = provider_profile['base_quality']
    quality_variance = random.uniform(-10, 15)  # Some randomness
    quality_score = max(1, min(100, base_quality + quality_variance))
    
    # Delivery time based on complexity and provider capability
    base_delivery = complexity * random.uniform(3, 8)
    delivery_reliability_factor = provider_profile['delivery_reliability']
    delivery_time = max(1, int(base_delivery / delivery_reliability_factor))
    
    # Relationship score based on past interactions (simulated)
    if provider_id == USER_PROVIDER_ID:
        # User provider has varying relationships
        relationship_score = random.uniform(0.3, 0.9)
    else:
        relationship_score = random.uniform(0.1, 0.8)
    
    # Experience with this client (affects pricing and quality)
    past_experience = random.choice(['None', 'Limited', 'Moderate', 'Extensive'])
    experience_multiplier = {'None': 1.0, 'Limited': 0.95, 'Moderate': 0.9, 'Extensive': 0.85}
    
    # Adjust price based on relationship and experience
    relationship_discount = relationship_score * 0.1  # Up to 10% discount for good relationships
    experience_discount = (1 - experience_multiplier[past_experience])
    
    final_price = base_price * (1 - relationship_discount - experience_discount)
    
    # Add market condition effects
    market_multiplier = {'Recession': 0.85, 'Recovery': 0.95, 'Growth': 1.05, 'Peak': 1.15}[market_conditions]
    final_price *= market_multiplier
    
    # Urgency level affects pricing
    urgency_level = random.choice(['Low', 'Medium', 'High', 'Critical'])
    urgency_premiums = {'Low': 1.0, 'Medium': 1.05, 'High': 1.15, 'Critical': 1.25}
    final_price *= urgency_premiums[urgency_level]
    
    # Additional bid characteristics
    technical_difficulty = random.randint(1, 10)
    proposal_quality = max(1, min(100, random.normalvariate(quality_score, 10)))
    team_experience = random.randint(1, provider_profile['years_experience'])
    innovation_score = max(1, min(100, provider_profile['innovation_focus'] * 100 + random.uniform(-20, 20)))
    
    # Risk assessment
    risk_factors = []
    if technical_difficulty > 7: risk_factors.append('Technical')
    if urgency_level in ['High', 'Critical']: risk_factors.append('Timeline')
    if relationship_score < 0.3: risk_factors.append('Relationship')
    
    risk_level = 'High' if len(risk_factors) >= 2 else ('Medium' if len(risk_factors) == 1 else 'Low')
    
    # Contract terms
    payment_terms = random.choice(['Net 30', 'Net 45', 'Net 60', '50% Upfront', 'Milestone-based'])
    contract_duration = random.randint(3, 24)  # months
    
    # Strategic factors
    incumbent_advantage = random.choice(['Yes', 'No'])
    strategic_importance = random.choice(['Low', 'Medium', 'High', 'Critical'])
    reference_strength = random.choice(['Weak', 'Moderate', 'Strong', 'Excellent'])
    
    return {
        'provider_id': provider_id,
        'price': final_price,
        'quality_score': round(quality_score, 1),
        'delivery_time': delivery_time,
        'relationship_score': round(relationship_score, 2),
        'urgency_level': urgency_level,
        'technical_difficulty': technical_difficulty,
        'past_experience': past_experience,
        'proposal_quality': round(proposal_quality, 1),
        'team_experience': team_experience,
        'innovation_score': round(innovation_score, 1),
        'risk_level': risk_level,
        'payment_terms': payment_terms,
        'contract_duration': contract_duration,
        'incumbent_advantage': incumbent_advantage,
        'strategic_importance': strategic_importance,
        'reference_strength': reference_strength
    }

def determine_winner(bids, client):
    """Determine winner using sophisticated multi-criteria scoring"""
    
    scored_bids = []
    
    for bid in bids:
        score = 0
        
        # Price score (lower is better, weighted by client price sensitivity)
        prices = [b['price'] for b in bids]
        min_price = min(prices)
        max_price = max(prices)
        if max_price > min_price:
            price_score = ((max_price - bid['price']) / (max_price - min_price)) * 100
        else:
            price_score = 100
        
        score += price_score * client['price_sensitivity']
        
        # Quality score (weighted by client quality focus)
        score += bid['quality_score'] * client['quality_focus']
        
        # Delivery time score (faster is better)
        delivery_times = [b['delivery_time'] for b in bids]
        min_delivery = min(delivery_times)
        max_delivery = max(delivery_times)
        if max_delivery > min_delivery:
            delivery_score = ((max_delivery - bid['delivery_time']) / (max_delivery - min_delivery)) * 100
        else:
            delivery_score = 100
        
        score += delivery_score * 0.3
        
        # Relationship score
        score += bid['relationship_score'] * client['relationship_importance'] * 50
        
        # Experience bonus
        experience_bonus = {'None': 0, 'Limited': 10, 'Moderate': 20, 'Extensive': 35}
        score += experience_bonus[bid['past_experience']]
        
        # Innovation score (if client values innovation)
        score += bid['innovation_score'] * client['innovation_appetite'] * 0.3
        
        # Proposal quality
        score += bid['proposal_quality'] * 0.2
        
        # Reference strength bonus
        reference_bonus = {'Weak': 0, 'Moderate': 5, 'Strong': 15, 'Excellent': 25}
        score += reference_bonus[bid['reference_strength']]
        
        # Incumbent advantage
        if bid['incumbent_advantage'] == 'Yes':
            score += 20
        
        # Strategic importance bonus
        strategic_bonus = {'Low': 0, 'Medium': 5, 'High': 15, 'Critical': 25}
        score += strategic_bonus[bid['strategic_importance']]
        
        # Risk penalty
        risk_penalty = {'Low': 0, 'Medium': -10, 'High': -25}
        score += risk_penalty[bid['risk_level']]
        
        # Add some randomness to simulate subjective factors
        score += random.uniform(-15, 15)
        
        scored_bids.append({'bid': bid, 'score': score})
    
    # Return the highest scoring bid
    winner = max(scored_bids, key=lambda x: x['score'])
    return winner['bid']

def calculate_competition_intensity(num_bids, project_value):
    """Calculate competition intensity score"""
    base_intensity = min(100, (num_bids - 2) * 20)  # More bidders = higher intensity
    
    # High-value projects tend to have more intense competition
    if project_value > 500000:
        base_intensity += 20
    elif project_value > 100000:
        base_intensity += 10
    
    return min(100, max(10, base_intensity))

def generate_industry_metadata(client_industry, project_value, complexity, geographic_region):
    """Generate industry-specific metadata based on industry type"""
    
    # Industry-specific material and cost data
    industry_data = {
        'Manufacturing': {
            'materials': ['Steel', 'Aluminum', 'Plastics', 'Composites', 'Rubber', 'Copper', 'Brass'],
            'material_cost_range': (0.15, 0.45),  # As percentage of project value
            'shipping_cost_range': (0.05, 0.15),
            'factories': ['Shenzhen, China', 'Guangzhou, China', 'Ho Chi Minh, Vietnam', 
                         'Tijuana, Mexico', 'Detroit, USA', 'Stuttgart, Germany'],
            'lead_time_range': (15, 90),
            'certifications': ['ISO 9001', 'ISO 14001', 'AS9100', 'IATF 16949'],
            'unit_sizes': ['10mm-50mm', '50mm-200mm', '200mm-500mm', '500mm-2000mm']
        },
        'Construction': {
            'materials': ['Concrete', 'Steel Beams', 'Lumber', 'Glass', 'Insulation', 
                         'Copper Wiring', 'PVC Piping', 'Drywall', 'Roofing Materials'],
            'material_cost_range': (0.35, 0.60),
            'shipping_cost_range': (0.03, 0.08),
            'factories': ['Local Supplier', 'Regional Hub', 'National Distributor'],
            'lead_time_range': (7, 45),
            'certifications': ['LEED', 'ISO 45001', 'OSHA Compliant', 'Green Building'],
            'unit_sizes': ['Residential (1000-5000 sqft)', 'Commercial (5000-50000 sqft)', 
                          'Industrial (50000+ sqft)']
        },
        'Technology': {
            'materials': ['Silicon Wafers', 'PCB Boards', 'Rare Earth Metals', 'Lithium Batteries',
                         'Copper Wiring', 'Gold Connectors', 'Aluminum Casings'],
            'material_cost_range': (0.25, 0.50),
            'shipping_cost_range': (0.02, 0.07),
            'factories': ['TSMC Taiwan', 'Samsung South Korea', 'Foxconn China', 
                         'Intel USA', 'Sony Japan'],
            'lead_time_range': (30, 120),
            'certifications': ['RoHS', 'CE Mark', 'FCC', 'UL Listed', 'Energy Star'],
            'unit_sizes': ['Chip Level', 'Board Level', 'Device Level', 'System Level']
        },
        'Healthcare': {
            'materials': ['Medical Grade Plastics', 'Stainless Steel 316L', 'Titanium Grade 5',
                         'Silicone', 'PEEK Polymer', 'Nitinol', 'Bioceramics'],
            'material_cost_range': (0.30, 0.55),
            'shipping_cost_range': (0.04, 0.10),
            'factories': ['Boston, USA', 'Munich, Germany', 'Zurich, Switzerland', 
                         'Tokyo, Japan', 'Dublin, Ireland'],
            'lead_time_range': (45, 180),
            'certifications': ['FDA Approved', 'CE Mark', 'ISO 13485', 'GMP', 'MDR Compliant'],
            'unit_sizes': ['Single Use', 'Multi-Use Device', 'Capital Equipment', 'Implantable']
        },
        'Automotive': {
            'materials': ['High-Strength Steel', 'Aluminum Alloys', 'Carbon Fiber', 'ABS Plastic',
                         'Rubber Compounds', 'Glass', 'Leather', 'Electronics'],
            'material_cost_range': (0.40, 0.65),
            'shipping_cost_range': (0.06, 0.12),
            'factories': ['Detroit, USA', 'Puebla, Mexico', 'Wolfsburg, Germany', 
                         'Nagoya, Japan', 'Shanghai, China', 'Seoul, South Korea'],
            'lead_time_range': (20, 90),
            'certifications': ['IATF 16949', 'ISO 14001', 'REACH', 'ELV Directive'],
            'unit_sizes': ['Components', 'Modules', 'Systems', 'Complete Vehicle']
        },
        'Energy': {
            'materials': ['Solar Panels', 'Wind Turbine Blades', 'Lithium Batteries', 
                         'Transformers', 'Power Cables', 'Inverters', 'Steel Structures'],
            'material_cost_range': (0.45, 0.70),
            'shipping_cost_range': (0.08, 0.18),
            'factories': ['Wuxi, China', 'Phoenix, USA', 'Hamburg, Germany', 
                         'Gujarat, India', 'Aalborg, Denmark'],
            'lead_time_range': (60, 240),
            'certifications': ['IEC 61215', 'UL 1741', 'IEEE 1547', 'ISO 50001'],
            'unit_sizes': ['Residential (1-10kW)', 'Commercial (10-500kW)', 
                          'Utility (500kW-100MW)']
        },
        'Retail': {
            'materials': ['Cardboard Packaging', 'Plastic Packaging', 'Display Materials',
                         'RFID Tags', 'Labels', 'Shelving Units'],
            'material_cost_range': (0.10, 0.25),
            'shipping_cost_range': (0.08, 0.20),
            'factories': ['Dongguan, China', 'Hanoi, Vietnam', 'Dhaka, Bangladesh', 
                         'Istanbul, Turkey'],
            'lead_time_range': (15, 60),
            'certifications': ['FSC Certified', 'Fair Trade', 'BSCI', 'WRAP'],
            'unit_sizes': ['Individual Item', 'Case Pack', 'Pallet', 'Container']
        },
        'Logistics': {
            'materials': ['Wooden Pallets', 'Plastic Pallets', 'Shipping Containers',
                         'Stretch Wrap', 'Tracking Devices', 'Dunnage'],
            'material_cost_range': (0.05, 0.15),
            'shipping_cost_range': (0.10, 0.25),
            'factories': ['Various Global Locations'],
            'lead_time_range': (5, 30),
            'certifications': ['ISPM 15', 'C-TPAT', 'ISO 28000', 'TAPA'],
            'unit_sizes': ['LTL', 'FTL', 'FCL', 'LCL']
        },
        'Aerospace': {
            'materials': ['Titanium Alloys', 'Carbon Fiber Composites', 'Aluminum 7075',
                         'Inconel', 'Kevlar', 'Specialty Ceramics', 'Hydraulic Fluids'],
            'material_cost_range': (0.50, 0.75),
            'shipping_cost_range': (0.05, 0.12),
            'factories': ['Seattle, USA', 'Toulouse, France', 'Bristol, UK', 
                         'Montreal, Canada', 'Sao Paulo, Brazil'],
            'lead_time_range': (90, 365),
            'certifications': ['AS9100', 'NADCAP', 'FAA Part 145', 'EASA Part 21'],
            'unit_sizes': ['Components', 'Sub-assemblies', 'Major Structures', 'Complete Aircraft']
        },
        'Finance': {
            'materials': ['Software Licenses', 'Cloud Infrastructure', 'Security Hardware',
                         'Network Equipment', 'Data Storage'],
            'material_cost_range': (0.20, 0.40),
            'shipping_cost_range': (0.01, 0.03),
            'factories': ['N/A - Digital Delivery'],
            'lead_time_range': (1, 30),
            'certifications': ['SOC 2', 'ISO 27001', 'PCI DSS', 'GDPR Compliant'],
            'unit_sizes': ['Per User', 'Department', 'Enterprise', 'Global']
        }
    }
    
    # Get industry-specific data or use default
    if client_industry not in industry_data:
        # Default values for unknown industries
        client_industry = 'Manufacturing'
    
    industry = industry_data[client_industry]
    
    # Select material type
    material_type = random.choice(industry['materials'])
    
    # Calculate material cost based on project value
    material_cost_pct = random.uniform(*industry['material_cost_range'])
    material_cost = project_value * material_cost_pct
    
    # Determine quantity and unit size based on project complexity
    if complexity <= 3:
        material_quantity = random.randint(10, 100)
        unit_size = industry['unit_sizes'][0] if industry['unit_sizes'] else 'Small'
    elif complexity <= 6:
        material_quantity = random.randint(100, 1000)
        unit_size = industry['unit_sizes'][1] if len(industry['unit_sizes']) > 1 else 'Medium'
    elif complexity <= 8:
        material_quantity = random.randint(1000, 10000)
        unit_size = industry['unit_sizes'][2] if len(industry['unit_sizes']) > 2 else 'Large'
    else:
        material_quantity = random.randint(10000, 100000)
        unit_size = industry['unit_sizes'][-1] if industry['unit_sizes'] else 'Extra Large'
    
    total_units = material_quantity
    
    # Select factory location
    factory_location = random.choice(industry['factories'])
    
    # Determine production capacity (units per month)
    production_capacity = material_quantity * random.uniform(1.5, 4.0)
    
    # Calculate shipping cost
    shipping_cost_pct = random.uniform(*industry['shipping_cost_range'])
    shipping_cost = project_value * shipping_cost_pct
    
    # Determine shipping method based on urgency and location
    if 'China' in factory_location or 'Asia' in geographic_region:
        shipping_methods = ['Sea Freight', 'Air Freight', 'Rail Freight']
    else:
        shipping_methods = ['Truck', 'Rail', 'Air Freight', 'Intermodal']
    
    shipping_method = random.choice(shipping_methods)
    
    # Lead time based on industry and shipping method
    base_lead_time = random.randint(*industry['lead_time_range'])
    if shipping_method == 'Sea Freight':
        lead_time_days = base_lead_time + random.randint(15, 30)
    elif shipping_method == 'Air Freight':
        lead_time_days = base_lead_time + random.randint(2, 7)
    else:
        lead_time_days = base_lead_time + random.randint(5, 15)
    
    # Warehousing cost (monthly)
    warehousing_cost = material_cost * random.uniform(0.01, 0.03)
    
    # Customs duties (for international shipments)
    if 'China' in factory_location or 'Vietnam' in factory_location or 'Mexico' in factory_location:
        customs_duties = material_cost * random.uniform(0.05, 0.25)
    else:
        customs_duties = 0
    
    # Quality certification
    quality_certification = random.choice(industry['certifications'])
    
    # Sustainability rating (1-10 scale)
    sustainability_rating = random.randint(3, 10)
    
    # Supplier tier (1 = direct OEM, 2 = major supplier, 3 = sub-supplier)
    supplier_tier = random.choice([1, 1, 2, 2, 2, 3, 3, 3, 3])
    
    return {
        'material_type': material_type,
        'material_cost': round(material_cost, 2),
        'material_quantity': material_quantity,
        'unit_size': unit_size,
        'total_units': total_units,
        'factory_location': factory_location,
        'production_capacity': round(production_capacity, 0),
        'shipping_cost': round(shipping_cost, 2),
        'shipping_method': shipping_method,
        'lead_time_days': lead_time_days,
        'warehousing_cost': round(warehousing_cost, 2),
        'customs_duties': round(customs_duties, 2),
        'quality_certification': quality_certification,
        'sustainability_rating': sustainability_rating,
        'supplier_tier': supplier_tier
    }

def write_to_csv(data, output_file):
    """Write data to CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Generated {len(data)} enhanced bid records in {output_file}")

def generate_analytics_summary(data):
    """Generate summary analytics for the dataset"""
    print("\n" + "="*80)
    print("ENHANCED BIDDING DATA ANALYTICS SUMMARY")
    print("="*80)
    
    # User provider performance
    user_bids = [record for record in data if record['provider_id'] == USER_PROVIDER_ID]
    user_wins = [record for record in user_bids if record['win_loss'] == 'win']
    
    print(f"\n🎯 YOUR PERFORMANCE ({USER_PROVIDER_ID}):")
    print(f"   Total Bids: {len(user_bids)}")
    print(f"   Wins: {len(user_wins)}")
    print(f"   Win Rate: {len(user_wins)/len(user_bids)*100:.1f}%")
    if user_bids:
        print(f"   Avg Bid Amount: ${np.mean([float(b['price']) for b in user_bids]):,.0f}")
        print(f"   Avg Quality Score: {np.mean([float(b['quality_score']) for b in user_bids]):.1f}")
    
    # Client analysis
    clients = {}
    for record in data:
        client = record['client_id']
        if client not in clients:
            clients[client] = {'total': 0, 'wins': 0, 'user_bids': 0, 'user_wins': 0}
        
        clients[client]['total'] += 1
        if record['win_loss'] == 'win':
            clients[client]['wins'] += 1
        
        if record['provider_id'] == USER_PROVIDER_ID:
            clients[client]['user_bids'] += 1
            if record['win_loss'] == 'win':
                clients[client]['user_wins'] += 1
    
    print(f"\n📊 CLIENT ANALYSIS:")
    print(f"   Total Clients: {len(clients)}")
    user_clients = [c for c in clients.values() if c['user_bids'] > 0]
    print(f"   Clients You've Bid For: {len(user_clients)}")
    
    # Competition analysis
    competitors = {}
    for record in data:
        provider = record['provider_id']
        if provider != USER_PROVIDER_ID:
            if provider not in competitors:
                competitors[provider] = {'bids': 0, 'wins': 0}
            competitors[provider]['bids'] += 1
            if record['win_loss'] == 'win':
                competitors[provider]['wins'] += 1
    
    print(f"\n🏆 TOP COMPETITORS (by win rate, min 10 bids):")
    top_competitors = []
    for provider, stats in competitors.items():
        if stats['bids'] >= 10:
            win_rate = stats['wins'] / stats['bids']
            top_competitors.append((provider, win_rate, stats['wins'], stats['bids']))
    
    top_competitors.sort(key=lambda x: x[1], reverse=True)
    for i, (provider, win_rate, wins, bids) in enumerate(top_competitors[:5]):
        print(f"   {i+1}. {provider}: {win_rate*100:.1f}% ({wins}/{bids})")
    
    # Project categories
    categories = {}
    for record in data:
        cat = record['project_category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    print(f"\n📈 PROJECT CATEGORIES:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"   {cat}: {count} bids")
    
    # Industry-specific metadata summary
    print(f"\n🏭 INDUSTRY-SPECIFIC INSIGHTS:")
    
    # Material types analysis
    materials = {}
    shipping_methods = {}
    factories = {}
    total_material_cost = 0
    total_shipping_cost = 0
    
    for record in data:
        # Materials
        material = record.get('material_type', 'Unknown')
        if material not in materials:
            materials[material] = 0
        materials[material] += 1
        
        # Shipping methods
        method = record.get('shipping_method', 'Unknown')
        if method not in shipping_methods:
            shipping_methods[method] = 0
        shipping_methods[method] += 1
        
        # Factory locations
        factory = record.get('factory_location', 'Unknown')
        if factory not in factories:
            factories[factory] = 0
        factories[factory] += 1
        
        # Costs
        total_material_cost += float(record.get('material_cost', 0))
        total_shipping_cost += float(record.get('shipping_cost', 0))
    
    print(f"   Average Material Cost: ${total_material_cost/len(data):,.0f}")
    print(f"   Average Shipping Cost: ${total_shipping_cost/len(data):,.0f}")
    
    print(f"\n   Top Materials Used:")
    for material, count in sorted(materials.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {material}: {count} projects")
    
    print(f"\n   Top Factory Locations:")
    for factory, count in sorted(factories.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {factory}: {count} projects")
    
    print(f"\n   Shipping Methods:")
    for method, count in sorted(shipping_methods.items(), key=lambda x: x[1], reverse=True):
        print(f"      {method}: {count} shipments")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("Generating enhanced B2B proposal bidding data...")
    print(f"User Provider ID: {USER_PROVIDER_ID}")
    print(f"Target Records: {num_records}")
    
    # Generate data
    bidding_data = generate_enhanced_bidding_data(num_records)
    
    # Sort by date
    bidding_data.sort(key=lambda x: x['date'])
    
    # Write to CSV
    write_to_csv(bidding_data, output_file)
    
    # Generate analytics summary
    generate_analytics_summary(bidding_data)
    
    print(f"\n✅ Data generation complete!")
    print(f"📁 File saved as: {output_file}")
    print(f"🎯 Ready to upload to your AI Proposal Optimization Platform!")
