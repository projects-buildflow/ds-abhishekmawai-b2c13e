"""
Buildly Data Generator - Production Quality

Generates large, realistic e-commerce datasets with intentional data quality
issues for students to discover and fix during their internship.

Features:
- Realistic customer profiles using Faker
- Proper statistical distributions for orders
- Time-series patterns (seasonality, trends)
- Controlled data quality issues (duplicates, nulls, outliers)
- Multiple related tables for complex analysis

Usage:
    pip install faker numpy
    python generate_data.py --customers 10000 --seed 42

Output Files:
    - customers.csv: Customer profiles with demographics
    - orders.csv: Transaction history
    - order_items.csv: Line items per order
    - products.csv: Product catalog
    - categories.csv: Product categories
    - marketing_campaigns.csv: Campaign performance
    - website_sessions.csv: User session data
    - customer_support.csv: Support tickets
"""

import argparse
import csv
import random
import math
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    print("Note: Install 'faker' for better data quality: pip install faker")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Note: Install 'numpy' for better distributions: pip install numpy")


# ============== Configuration ==============

# Regional settings (controls name/email patterns)
REGIONS = {
    "India": {"weight": 0.50, "locale": "en_IN", "currency": "INR"},
    "USA": {"weight": 0.20, "locale": "en_US", "currency": "USD"},
    "UK": {"weight": 0.10, "locale": "en_GB", "currency": "GBP"},
    "Germany": {"weight": 0.05, "locale": "de_DE", "currency": "EUR"},
    "Canada": {"weight": 0.05, "locale": "en_CA", "currency": "CAD"},
    "Australia": {"weight": 0.05, "locale": "en_AU", "currency": "AUD"},
    "Singapore": {"weight": 0.05, "locale": "en_US", "currency": "SGD"},
}

# Product catalog (comprehensive)
CATEGORIES = [
    {"id": "CAT01", "name": "Electronics", "margin": 0.25},
    {"id": "CAT02", "name": "Office Furniture", "margin": 0.35},
    {"id": "CAT03", "name": "Accessories", "margin": 0.45},
    {"id": "CAT04", "name": "Audio", "margin": 0.30},
    {"id": "CAT05", "name": "Storage", "margin": 0.28},
    {"id": "CAT06", "name": "Lighting", "margin": 0.40},
    {"id": "CAT07", "name": "Ergonomics", "margin": 0.38},
    {"id": "CAT08", "name": "Cables & Adapters", "margin": 0.50},
]

PRODUCTS = [
    # Electronics
    {"id": "P1001", "name": "Wireless Mouse Pro", "category_id": "CAT01", "base_price": 599, "cost": 450},
    {"id": "P1002", "name": "Wireless Mouse Basic", "category_id": "CAT01", "base_price": 299, "cost": 180},
    {"id": "P1003", "name": "Mechanical Keyboard RGB", "category_id": "CAT01", "base_price": 2499, "cost": 1500},
    {"id": "P1004", "name": "Membrane Keyboard", "category_id": "CAT01", "base_price": 799, "cost": 400},
    {"id": "P1005", "name": "USB-C Hub 7-in-1", "category_id": "CAT01", "base_price": 1299, "cost": 700},
    {"id": "P1006", "name": "USB-C Hub 4-in-1", "category_id": "CAT01", "base_price": 699, "cost": 350},
    {"id": "P1007", "name": "4K Webcam", "category_id": "CAT01", "base_price": 3999, "cost": 2800},
    {"id": "P1008", "name": "HD Webcam", "category_id": "CAT01", "base_price": 1499, "cost": 900},
    {"id": "P1009", "name": "Portable Monitor 15.6\"", "category_id": "CAT01", "base_price": 12999, "cost": 9500},
    {"id": "P1010", "name": "Wireless Charger Pad", "category_id": "CAT01", "base_price": 899, "cost": 400},

    # Office Furniture
    {"id": "P2001", "name": "Ergonomic Office Chair", "category_id": "CAT02", "base_price": 15999, "cost": 10000},
    {"id": "P2002", "name": "Standing Desk Electric", "category_id": "CAT02", "base_price": 24999, "cost": 16000},
    {"id": "P2003", "name": "Monitor Stand Wooden", "category_id": "CAT02", "base_price": 1499, "cost": 800},
    {"id": "P2004", "name": "Monitor Stand Metal", "category_id": "CAT02", "base_price": 999, "cost": 500},
    {"id": "P2005", "name": "Desk Organizer Set", "category_id": "CAT02", "base_price": 799, "cost": 350},
    {"id": "P2006", "name": "Filing Cabinet 3-Drawer", "category_id": "CAT02", "base_price": 4999, "cost": 3000},

    # Accessories
    {"id": "P3001", "name": "Laptop Sleeve 15\"", "category_id": "CAT03", "base_price": 999, "cost": 400},
    {"id": "P3002", "name": "Laptop Backpack Pro", "category_id": "CAT03", "base_price": 2499, "cost": 1200},
    {"id": "P3003", "name": "Mouse Pad XL Gaming", "category_id": "CAT03", "base_price": 599, "cost": 200},
    {"id": "P3004", "name": "Mouse Pad Standard", "category_id": "CAT03", "base_price": 199, "cost": 50},
    {"id": "P3005", "name": "Wrist Rest Keyboard", "category_id": "CAT03", "base_price": 699, "cost": 300},
    {"id": "P3006", "name": "Blue Light Glasses", "category_id": "CAT03", "base_price": 999, "cost": 350},
    {"id": "P3007", "name": "Laptop Stand Adjustable", "category_id": "CAT03", "base_price": 1299, "cost": 600},

    # Audio
    {"id": "P4001", "name": "Noise Cancelling Headphones", "category_id": "CAT04", "base_price": 7999, "cost": 5500},
    {"id": "P4002", "name": "Wireless Earbuds Pro", "category_id": "CAT04", "base_price": 4999, "cost": 3000},
    {"id": "P4003", "name": "Wireless Earbuds Basic", "category_id": "CAT04", "base_price": 1999, "cost": 1000},
    {"id": "P4004", "name": "USB Microphone Podcast", "category_id": "CAT04", "base_price": 5999, "cost": 3500},
    {"id": "P4005", "name": "Desktop Speakers 2.1", "category_id": "CAT04", "base_price": 3499, "cost": 2000},

    # Storage
    {"id": "P5001", "name": "Portable SSD 1TB", "category_id": "CAT05", "base_price": 5999, "cost": 4500},
    {"id": "P5002", "name": "Portable SSD 500GB", "category_id": "CAT05", "base_price": 3499, "cost": 2500},
    {"id": "P5003", "name": "USB Flash Drive 128GB", "category_id": "CAT05", "base_price": 699, "cost": 350},
    {"id": "P5004", "name": "External HDD 2TB", "category_id": "CAT05", "base_price": 4499, "cost": 3200},
    {"id": "P5005", "name": "Memory Card 256GB", "category_id": "CAT05", "base_price": 1999, "cost": 1200},

    # Lighting
    {"id": "P6001", "name": "LED Desk Lamp Smart", "category_id": "CAT06", "base_price": 2499, "cost": 1400},
    {"id": "P6002", "name": "LED Desk Lamp Basic", "category_id": "CAT06", "base_price": 899, "cost": 400},
    {"id": "P6003", "name": "Monitor Light Bar", "category_id": "CAT06", "base_price": 1999, "cost": 1000},
    {"id": "P6004", "name": "Ring Light 12\"", "category_id": "CAT06", "base_price": 1499, "cost": 700},

    # Ergonomics
    {"id": "P7001", "name": "Standing Desk Mat", "category_id": "CAT07", "base_price": 2999, "cost": 1800},
    {"id": "P7002", "name": "Footrest Adjustable", "category_id": "CAT07", "base_price": 1499, "cost": 800},
    {"id": "P7003", "name": "Laptop Cooling Pad", "category_id": "CAT07", "base_price": 999, "cost": 500},
    {"id": "P7004", "name": "Document Holder", "category_id": "CAT07", "base_price": 599, "cost": 250},

    # Cables & Adapters
    {"id": "P8001", "name": "USB-C Cable 2m Braided", "category_id": "CAT08", "base_price": 499, "cost": 150},
    {"id": "P8002", "name": "HDMI Cable 4K 2m", "category_id": "CAT08", "base_price": 599, "cost": 200},
    {"id": "P8003", "name": "Cable Management Kit", "category_id": "CAT08", "base_price": 799, "cost": 300},
    {"id": "P8004", "name": "Power Strip 6-Outlet", "category_id": "CAT08", "base_price": 999, "cost": 450},
    {"id": "P8005", "name": "USB-C to HDMI Adapter", "category_id": "CAT08", "base_price": 899, "cost": 400},
]

# Order statuses with transition probabilities
ORDER_STATUSES = {
    "pending": {"weight": 0.05, "can_transition": ["processing", "cancelled"]},
    "processing": {"weight": 0.08, "can_transition": ["shipped", "cancelled"]},
    "shipped": {"weight": 0.10, "can_transition": ["delivered", "returned"]},
    "delivered": {"weight": 0.70, "can_transition": ["returned"]},
    "cancelled": {"weight": 0.04, "can_transition": []},
    "returned": {"weight": 0.03, "can_transition": []},
}

# Payment methods
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet", "COD"]
PAYMENT_WEIGHTS = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]


# ============== Data Generators ==============

class DataGenerator:
    """Main data generator class."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        if NUMPY_AVAILABLE:
            np.random.seed(seed)

        # Initialize Faker with multiple locales
        if FAKER_AVAILABLE:
            self.fakers = {}
            for region, config in REGIONS.items():
                self.fakers[region] = Faker(config["locale"])
                self.fakers[region].seed_instance(seed)
        else:
            self.fakers = None

        # Track generated data for relationships
        self.customers = []
        self.customer_emails = {}  # For duplicate detection
        self.orders = []
        self.order_items = []

    def _weighted_choice(self, choices: list, weights: list):
        """Make a weighted random choice."""
        if NUMPY_AVAILABLE:
            return np.random.choice(choices, p=weights)
        else:
            return random.choices(choices, weights=weights)[0]

    def _pareto_int(self, alpha: float = 1.5, max_val: int = 50) -> int:
        """Generate integer from Pareto distribution (for realistic order counts)."""
        if NUMPY_AVAILABLE:
            val = int(np.random.pareto(alpha))
        else:
            val = int(random.paretovariate(alpha))
        return min(val, max_val)

    def _generate_name(self, region: str) -> tuple:
        """Generate a realistic name for a region."""
        if self.fakers:
            faker = self.fakers[region]
            return faker.first_name(), faker.last_name()
        else:
            # Fallback names — large pools per region to avoid duplicate emails
            _FALLBACK_FIRST = {
                "India": ["Rahul", "Priya", "Amit", "Sneha", "Arjun", "Meera", "Vikram", "Ananya",
                           "Rohit", "Kavya", "Nikhil", "Divya", "Saurabh", "Pooja", "Aditya", "Neha",
                           "Kartik", "Ishita", "Manish", "Shruti", "Deepak", "Riya", "Varun", "Tanvi",
                           "Harsh", "Pallavi", "Gaurav", "Swati", "Rajesh", "Komal", "Akash", "Sakshi",
                           "Vishal", "Preeti", "Kunal", "Aarti", "Siddharth", "Megha", "Ashish", "Nisha"],
                "USA": ["James", "Mary", "Robert", "Jennifer", "Michael", "Linda", "David", "Sarah",
                         "William", "Jessica", "Daniel", "Emily", "Joseph", "Hannah", "Andrew", "Megan",
                         "Tyler", "Rachel", "Brandon", "Nicole", "Nathan", "Ashley", "Ryan", "Samantha",
                         "Kevin", "Lauren", "Brian", "Amanda", "Justin", "Stephanie", "Jason", "Brittany",
                         "Eric", "Heather", "Patrick", "Tiffany", "Sean", "Christina", "Jeremy", "Michelle"],
                "UK": ["Oliver", "Amelia", "Jack", "Olivia", "Harry", "Isla", "George", "Emily",
                        "Charlie", "Sophia", "Thomas", "Grace", "James", "Lily", "William", "Ella",
                        "Edward", "Charlotte", "Henry", "Freya", "Sam", "Chloe", "Ben", "Mia",
                        "Alexander", "Poppy", "Daniel", "Evie", "Luke", "Ruby", "Matthew", "Daisy",
                        "Ethan", "Alice", "Noah", "Isabella", "Leo", "Jessica", "Oscar", "Florence"],
                "Germany": ["Lukas", "Emma", "Leon", "Mia", "Finn", "Hannah", "Jonas", "Lina",
                             "Felix", "Emilia", "Paul", "Marie", "Max", "Sophie", "Elias", "Anna",
                             "Noah", "Lea", "Ben", "Clara", "Liam", "Lena", "Tim", "Johanna",
                             "Jan", "Laura", "Nico", "Sarah", "Tom", "Julia", "Moritz", "Lisa",
                             "David", "Amelie", "Julian", "Maja", "Erik", "Nele", "Philipp", "Katharina"],
                "Canada": ["Liam", "Emma", "Noah", "Olivia", "Ethan", "Ava", "Lucas", "Sophia",
                            "Mason", "Isabella", "Logan", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
                            "Jacob", "Harper", "Alexander", "Evelyn", "Daniel", "Abigail", "Henry", "Emily",
                            "Sebastian", "Ella", "Jack", "Elizabeth", "Owen", "Camila", "Dylan", "Luna",
                            "Nathan", "Sofia", "Caleb", "Avery", "Ryan", "Scarlett", "Luke", "Grace"],
                "Australia": ["Oliver", "Charlotte", "William", "Amelia", "Jack", "Olivia", "Noah", "Isla",
                               "Thomas", "Mia", "Leo", "Ava", "Charlie", "Grace", "Henry", "Willow",
                               "James", "Harper", "Liam", "Ella", "Ethan", "Chloe", "Lucas", "Zoe",
                               "Mason", "Lily", "Alexander", "Ruby", "Daniel", "Sophie", "Samuel", "Ivy",
                               "Benjamin", "Matilda", "Max", "Sienna", "Archie", "Evie", "Oscar", "Layla"],
                "Singapore": ["Wei", "Mei", "Jun", "Xin", "Kai", "Hui", "Zhi", "Ying",
                               "Jia", "Lin", "Chen", "Fang", "Ming", "Rui", "Hong", "Yan",
                               "Shen", "Wen", "Hao", "Qi", "Raj", "Priya", "Arun", "Lakshmi",
                               "Ahmad", "Siti", "Muhammad", "Nur", "Ismail", "Aminah", "Ali", "Fatimah",
                               "Daniel", "Sarah", "Ryan", "Rachel", "Marcus", "Nicole", "Joshua", "Hannah"],
            }
            _FALLBACK_LAST = {
                "India": ["Kumar", "Singh", "Patel", "Sharma", "Gupta", "Nair", "Reddy", "Iyer",
                           "Verma", "Joshi", "Shah", "Mehta", "Das", "Rao", "Chatterjee", "Mishra",
                           "Deshmukh", "Kapoor", "Bhat", "Pillai"],
                "USA": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                         "Rodriguez", "Martinez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson",
                         "Martin", "Lee", "Thompson", "White"],
                "UK": ["Smith", "Jones", "Taylor", "Brown", "Williams", "Wilson", "Johnson", "Davies",
                        "Robinson", "Wright", "Thompson", "Evans", "Walker", "White", "Roberts", "Green",
                        "Hall", "Wood", "Jackson", "Clarke"],
                "Germany": ["Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker",
                             "Schulz", "Hoffmann", "Koch", "Richter", "Wolf", "Schroeder", "Neumann", "Schwarz",
                             "Zimmermann", "Braun", "Krueger", "Hartmann"],
                "Canada": ["Smith", "Brown", "Tremblay", "Martin", "Roy", "Wilson", "MacDonald", "Taylor",
                            "Campbell", "Anderson", "Jones", "Leblanc", "Johnson", "Williams", "Thomson", "Cote",
                            "Singh", "Patel", "Lee", "Chen"],
                "Australia": ["Smith", "Jones", "Williams", "Brown", "Wilson", "Taylor", "Johnson", "White",
                               "Martin", "Anderson", "Thompson", "Nguyen", "Thomas", "Walker", "Harris", "Lee",
                               "Ryan", "Robinson", "Kelly", "King"],
                "Singapore": ["Tan", "Lim", "Lee", "Ng", "Ong", "Wong", "Goh", "Chua",
                               "Chan", "Koh", "Teo", "Ang", "Yeo", "Ho", "Sim", "Low",
                               "Kumar", "Singh", "Ahmad", "Mohamed"],
            }
            first_names = _FALLBACK_FIRST.get(region, _FALLBACK_FIRST["USA"])
            last_names = _FALLBACK_LAST.get(region, _FALLBACK_LAST["USA"])
            return random.choice(first_names), random.choice(last_names)

    def _generate_email(self, first_name: str, last_name: str, variation: str = None) -> str:
        """Generate email with optional case/format variations for duplicates."""
        first = first_name.lower().replace(" ", "")
        last = last_name.lower().replace(" ", "")
        domains = ["gmail.com", "yahoo.com", "outlook.com", "email.com", "hotmail.com"]
        domain = random.choice(domains)

        patterns = [
            f"{first}.{last}@{domain}",
            f"{first}{last}@{domain}",
            f"{first}.{last[0]}@{domain}",
            f"{first}_{last}@{domain}",
            f"{first}{random.randint(1, 99)}@{domain}",
        ]
        email = random.choice(patterns)

        if variation == "uppercase":
            return email.upper()
        elif variation == "mixed":
            return ''.join(c.upper() if random.random() > 0.5 else c for c in email)
        elif variation == "typo":
            # Common email typos
            email = email.replace("@gmail.com", "@gmial.com")
        return email

    def _random_date(self, start: datetime, end: datetime) -> datetime:
        """Generate random date between start and end."""
        delta = end - start
        random_days = random.randint(0, delta.days)
        return start + timedelta(days=random_days)

    def generate_customers(self, num_customers: int) -> list[dict]:
        """Generate customer data with realistic patterns and controlled data issues."""
        print(f"Generating {num_customers:,} customers...")

        customers = []
        regions = list(REGIONS.keys())
        region_weights = [REGIONS[r]["weight"] for r in regions]

        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 10, 31)

        # Customer segments (affects behavior)
        segments = ["New", "Active", "VIP", "At Risk", "Churned"]
        segment_weights = [0.25, 0.40, 0.15, 0.12, 0.08]

        for i in range(num_customers):
            customer_id = 10000 + i + 1
            region = self._weighted_choice(regions, region_weights)
            first_name, last_name = self._generate_name(region)

            # Decide if this is a duplicate (~3% duplicates)
            is_duplicate = random.random() < 0.03 and len(self.customer_emails) > 100
            if is_duplicate:
                # Pick an existing customer and reuse their name with an email case variation
                orig_email = random.choice(list(self.customer_emails.keys()))
                variation = random.choice(["uppercase", "mixed"])
                if variation == "uppercase":
                    email = orig_email.upper()
                else:
                    email = ''.join(c.upper() if random.random() > 0.5 else c for c in orig_email)
            else:
                email = self._generate_email(first_name, last_name)

            self.customer_emails[email.lower()] = customer_id

            signup_date = self._random_date(start_date, end_date)
            segment = self._weighted_choice(segments, segment_weights)

            # Age distribution (18-70, weighted toward 25-45)
            if NUMPY_AVAILABLE:
                age = int(np.clip(np.random.normal(35, 12), 18, 70))
            else:
                age = random.randint(22, 55)

            # Gender
            gender = self._weighted_choice(["M", "F", "Other", ""], [0.48, 0.48, 0.02, 0.02])

            customer = {
                "customer_id": customer_id,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone": self.fakers[region].phone_number() if self.fakers else f"+91-{random.randint(7000000000, 9999999999)}",
                "age": age,
                "gender": gender,
                "country": region,
                "city": self.fakers[region].city() if self.fakers else "Mumbai",
                "signup_date": signup_date.strftime("%Y-%m-%d"),
                "signup_source": self._weighted_choice(
                    ["Organic", "Paid Search", "Social", "Referral", "Email"],
                    [0.35, 0.25, 0.20, 0.12, 0.08]
                ),
                "segment": segment,
                "is_subscribed": self._weighted_choice([True, False], [0.65, 0.35]),
                "total_orders": 0,  # Updated after orders generated
                "total_spent": 0.0,  # Updated after orders generated
                "avg_order_value": 0.0,
                "last_order_date": None,
            }

            # Introduce data quality issues

            # Missing email (~1.5%)
            if random.random() < 0.015:
                customer["email"] = ""

            # Invalid email format (~0.8%)
            if random.random() < 0.008:
                customer["email"] = "not-valid-email"

            # Missing phone (~5%)
            if random.random() < 0.05:
                customer["phone"] = ""

            # Invalid age (~0.5%)
            if random.random() < 0.005:
                customer["age"] = random.choice([-1, 0, 150, 999])

            # Future signup date (~0.3%)
            if random.random() < 0.003:
                customer["signup_date"] = "2025-12-31"

            # Invalid date format (~0.2%)
            if random.random() < 0.002:
                customer["signup_date"] = "invalid-date"

            customers.append(customer)

            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1:,} customers...")

        self.customers = customers
        return customers

    def generate_orders(self, orders_per_customer_avg: float = 4.0) -> tuple[list, list]:
        """Generate orders and order items."""
        print(f"Generating orders (avg {orders_per_customer_avg} per customer)...")

        orders = []
        order_items = []
        order_id = 100000

        product_ids = [p["id"] for p in PRODUCTS]
        product_prices = {p["id"]: p["base_price"] for p in PRODUCTS}
        status_list = list(ORDER_STATUSES.keys())
        status_weights = [ORDER_STATUSES[s]["weight"] for s in status_list]

        for customer in self.customers:
            # Skip customers with invalid data
            try:
                signup = datetime.strptime(customer["signup_date"], "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            # Number of orders based on segment
            segment_multiplier = {
                "VIP": 3.0, "Active": 1.5, "New": 0.5, "At Risk": 0.8, "Churned": 0.3
            }
            multiplier = segment_multiplier.get(customer["segment"], 1.0)
            num_orders = max(0, self._pareto_int(alpha=2.0, max_val=30))
            num_orders = int(num_orders * multiplier)

            customer_total = 0.0
            last_order = None

            for _ in range(num_orders):
                order_id += 1

                # Order date after signup, with recency bias
                days_since_signup = (datetime(2024, 10, 31) - signup).days
                if days_since_signup <= 0:
                    continue

                # More recent orders more likely
                if NUMPY_AVAILABLE:
                    order_day = int(np.random.beta(2, 5) * days_since_signup)
                else:
                    order_day = random.randint(0, days_since_signup)

                order_date = signup + timedelta(days=order_day)

                # Seasonal adjustment — Q4 generally has more orders
                # except October 2024 which has an anomaly: fewer orders, extra cancellations, lower values
                is_october_anomaly = (order_date.year == 2024 and order_date.month == 10)

                # October 2024 anomaly: reduced order volume + extra cancellations
                if is_october_anomaly and random.random() < 0.15:
                    continue

                status = self._weighted_choice(status_list, status_weights)
                if is_october_anomaly and random.random() < 0.05:
                    status = "cancelled"
                payment = self._weighted_choice(PAYMENT_METHODS, PAYMENT_WEIGHTS)

                # Generate order items (1-5 items per order)
                num_items = self._weighted_choice([1, 2, 3, 4, 5], [0.45, 0.30, 0.15, 0.07, 0.03])
                order_products = random.sample(product_ids, min(num_items, len(product_ids)))

                order_total = 0.0
                order_items_list = []

                for item_idx, prod_id in enumerate(order_products):
                    quantity = self._weighted_choice([1, 2, 3], [0.70, 0.25, 0.05])
                    base_price = product_prices[prod_id]

                    # Price variation (discounts, etc.)
                    price_multiplier = 1.0
                    if random.random() < 0.20:  # 20% chance of discount
                        price_multiplier = random.uniform(0.85, 0.95)
                    # October 2024 anomaly: slightly lower order values
                    if is_october_anomaly:
                        price_multiplier *= random.uniform(0.90, 0.97)
                    unit_price = round(base_price * price_multiplier, 2)
                    item_total = unit_price * quantity

                    order_item = {
                        "order_item_id": f"{order_id}-{item_idx + 1}",
                        "order_id": order_id,
                        "product_id": prod_id,
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "discount_percent": round((1 - price_multiplier) * 100, 1) if price_multiplier < 1 else 0,
                        "item_total": round(item_total, 2),
                    }

                    # Data quality issues for order items
                    if random.random() < 0.005:  # 0.5% negative quantity
                        order_item["quantity"] = -1

                    order_items_list.append(order_item)
                    order_total += item_total

                # Shipping and tax
                shipping = 0 if order_total >= 999 else random.choice([49, 99, 149])
                tax_rate = 0.18  # GST
                tax = round(order_total * tax_rate, 2)
                grand_total = round(order_total + shipping + tax, 2)

                order = {
                    "order_id": order_id,
                    "customer_id": customer["customer_id"],
                    "order_date": order_date.strftime("%Y-%m-%d"),
                    "order_time": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00",
                    "status": status,
                    "payment_method": payment,
                    "subtotal": round(order_total, 2),
                    "shipping": shipping,
                    "tax": tax,
                    "total": grand_total,
                    "items_count": len(order_products),
                    "shipping_city": customer["city"],
                    "shipping_country": customer["country"],
                }

                # Data quality issues for orders
                if random.random() < 0.008:  # Future date
                    order["order_date"] = "2025-06-15"

                if random.random() < 0.005:  # Negative total
                    order["total"] = -grand_total

                if random.random() < 0.003:  # Missing customer_id
                    order["customer_id"] = None

                orders.append(order)
                order_items.extend(order_items_list)

                customer_total += grand_total
                last_order = order_date

            # Update customer aggregates
            customer["total_orders"] = num_orders
            customer["total_spent"] = round(customer_total, 2)
            customer["avg_order_value"] = round(customer_total / num_orders, 2) if num_orders > 0 else 0
            customer["last_order_date"] = last_order.strftime("%Y-%m-%d") if last_order else None

        print(f"  Generated {len(orders):,} orders with {len(order_items):,} line items")
        self.orders = orders
        self.order_items = order_items
        return orders, order_items

    def generate_marketing_campaigns(self, num_campaigns: int = 200) -> list[dict]:
        """Generate marketing campaign data for analysis."""
        print(f"Generating {num_campaigns} marketing campaigns...")

        campaigns = []
        channels = ["Facebook", "Instagram", "Google Ads", "Email", "Twitter", "YouTube", "LinkedIn", "TikTok"]
        campaign_types = ["Awareness", "Acquisition", "Retention", "Reactivation", "Seasonal"]

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 10, 31)

        for i in range(num_campaigns):
            campaign_start = self._random_date(start_date, end_date - timedelta(days=30))
            duration = random.randint(7, 60)
            campaign_end = campaign_start + timedelta(days=duration)

            channel = random.choice(channels)
            camp_type = random.choice(campaign_types)

            # Budget and spend
            budget = random.randint(5000, 500000)
            spend = budget * random.uniform(0.7, 1.0)

            # Performance metrics (with channel-specific patterns)
            base_ctr = {"Facebook": 0.9, "Instagram": 1.2, "Google Ads": 2.0, "Email": 3.5,
                        "Twitter": 0.5, "YouTube": 0.4, "LinkedIn": 0.8, "TikTok": 1.5}
            ctr = base_ctr.get(channel, 1.0) * random.uniform(0.5, 2.0) / 100

            impressions = int(spend * random.uniform(100, 500))
            clicks = int(impressions * ctr)
            conversions = int(clicks * random.uniform(0.01, 0.10))
            revenue = conversions * random.uniform(500, 5000)

            campaign = {
                "campaign_id": f"CAMP{10000 + i}",
                "campaign_name": f"{camp_type} - {channel} - {campaign_start.strftime('%b %Y')}",
                "channel": channel,
                "campaign_type": camp_type,
                "start_date": campaign_start.strftime("%Y-%m-%d"),
                "end_date": campaign_end.strftime("%Y-%m-%d"),
                "budget": round(budget, 2),
                "spend": round(spend, 2),
                "impressions": impressions,
                "clicks": clicks,
                "ctr": round(clicks / impressions * 100, 2) if impressions > 0 else 0,
                "conversions": conversions,
                "conversion_rate": round(conversions / clicks * 100, 2) if clicks > 0 else 0,
                "revenue": round(revenue, 2),
                "roas": round(revenue / spend, 2) if spend > 0 else 0,
                "cpa": round(spend / conversions, 2) if conversions > 0 else 0,
            }

            # Data issues
            if random.random() < 0.02:
                campaign["spend"] = budget * 1.5  # Overspend

            if random.random() < 0.01:
                campaign["impressions"] = -1000  # Invalid

            campaigns.append(campaign)

        return campaigns

    def generate_website_sessions(self, num_sessions: int = 50000) -> list[dict]:
        """Generate website session data for behavioral analysis."""
        print(f"Generating {num_sessions:,} website sessions...")

        sessions = []
        devices = ["Desktop", "Mobile", "Tablet"]
        device_weights = [0.35, 0.55, 0.10]
        browsers = ["Chrome", "Safari", "Firefox", "Edge", "Other"]
        browser_weights = [0.60, 0.20, 0.10, 0.07, 0.03]

        sources = ["Direct", "Organic Search", "Paid Search", "Social", "Email", "Referral"]
        source_weights = [0.25, 0.30, 0.20, 0.12, 0.08, 0.05]

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 10, 31)

        customer_ids = [c["customer_id"] for c in self.customers]

        for i in range(num_sessions):
            session_date = self._random_date(start_date, end_date)
            hour_weights = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05,
                           0.06, 0.08, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06,
                           0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.02, 0.03]  # Sums to 1.0
            hour = self._weighted_choice(list(range(24)), hour_weights)

            # 60% sessions are from existing customers
            if random.random() < 0.60:
                customer_id = random.choice(customer_ids)
            else:
                customer_id = None

            pages_viewed = self._pareto_int(alpha=2.5, max_val=30) + 1
            time_on_site = int(pages_viewed * random.uniform(20, 120))  # seconds

            session = {
                "session_id": f"S{1000000 + i}",
                "customer_id": customer_id,
                "session_date": session_date.strftime("%Y-%m-%d"),
                "session_hour": hour,
                "device": self._weighted_choice(devices, device_weights),
                "browser": self._weighted_choice(browsers, browser_weights),
                "traffic_source": self._weighted_choice(sources, source_weights),
                "landing_page": random.choice(["/", "/products", "/category", "/sale", "/new-arrivals"]),
                "pages_viewed": pages_viewed,
                "time_on_site_seconds": time_on_site,
                "bounced": pages_viewed == 1,
                "converted": random.random() < (0.03 if pages_viewed > 3 else 0.005),
            }

            sessions.append(session)

            if (i + 1) % 50000 == 0:
                print(f"  Generated {i + 1:,} sessions...")

        return sessions

    def generate_support_tickets(self, num_tickets: int = 5000) -> list[dict]:
        """Generate customer support ticket data."""
        print(f"Generating {num_tickets:,} support tickets...")

        tickets = []
        categories = ["Order Issue", "Product Query", "Returns", "Payment", "Shipping", "Technical", "Other"]
        category_weights = [0.25, 0.20, 0.15, 0.15, 0.12, 0.08, 0.05]

        priorities = ["Low", "Medium", "High", "Urgent"]
        priority_weights = [0.30, 0.45, 0.20, 0.05]

        statuses = ["Open", "In Progress", "Resolved", "Closed"]
        status_weights = [0.10, 0.15, 0.35, 0.40]

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 10, 31)

        customer_ids = [c["customer_id"] for c in self.customers if c["total_orders"] > 0]

        for i in range(num_tickets):
            created = self._random_date(start_date, end_date)
            customer_id = random.choice(customer_ids)
            category = self._weighted_choice(categories, category_weights)
            priority = self._weighted_choice(priorities, priority_weights)
            status = self._weighted_choice(statuses, status_weights)

            # Resolution time based on priority and status
            if status in ["Resolved", "Closed"]:
                base_hours = {"Low": 48, "Medium": 24, "High": 8, "Urgent": 2}
                resolution_hours = int(base_hours[priority] * random.uniform(0.5, 3.0))
                resolved_date = created + timedelta(hours=resolution_hours)
            else:
                resolved_date = None
                resolution_hours = None

            ticket = {
                "ticket_id": f"TKT{100000 + i}",
                "customer_id": customer_id,
                "created_date": created.strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "priority": priority,
                "status": status,
                "resolved_date": resolved_date.strftime("%Y-%m-%d %H:%M:%S") if resolved_date else None,
                "resolution_hours": resolution_hours,
                "satisfaction_score": random.randint(1, 5) if status == "Closed" else None,
                "agent_id": f"AGT{random.randint(1, 20):03d}",
            }

            tickets.append(ticket)

        return tickets

    def generate_marketing_customers_raw(self) -> list[dict]:
        """Generate a messy 'marketing export' of customer data.

        Simulates a CSV dump from a marketing tool with different column names,
        extra junk columns, more missing values, and whitespace padding.
        """
        print("Generating marketing_customers_raw (messy export)...")

        raw_records = []
        sources = ["facebook_lead", "google_form", "webinar_signup", "email_capture", "partner_import"]

        for c in self.customers:
            # ~10% missing values across various fields
            name = f"  {c['first_name']} {c['last_name']}  " if random.random() > 0.05 else ""
            email = c["email"] if random.random() > 0.08 else ""
            age = c["age"] if random.random() > 0.10 else ""
            phone = c["phone"] if random.random() > 0.12 else ""

            # Whitespace padding on some fields
            city = f" {c['city']} " if random.random() > 0.3 else c["city"]

            record = {
                "full_name": name,
                "email_address": email,
                "age": age,
                "gender": c["gender"],
                "phone_number": phone,
                "location": city,
                "country": c["country"],
                "date_joined": c["signup_date"],
                "lead_source": random.choice(sources),
                "utm_campaign": f"camp_{random.randint(100, 999)}" if random.random() > 0.4 else "",
                "utm_medium": random.choice(["cpc", "organic", "social", "email", ""]),
                "notes": random.choice(["", "", "", "follow up", "VIP", "do not contact", ""]),
                "is_subscribed": c["is_subscribed"],
            }
            raw_records.append(record)

        print(f"  Generated {len(raw_records):,} marketing records")
        return raw_records


# ============== Output Functions ==============

def save_csv(data: list[dict], filepath: Path, fieldnames: list = None):
    """Save data to CSV file."""
    if not data:
        print(f"  Warning: No data to save for {filepath.name}")
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"  Saved {filepath.name}: {len(data):,} records")


def main():
    parser = argparse.ArgumentParser(description="Generate Buildly e-commerce dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--customers", type=int, default=10000, help="Number of customers")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument("--skip-sessions", action="store_true", help="Skip generating sessions (large file)")

    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 60)
    print("BUILDLY DATA GENERATOR")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Customers: {args.customers:,}")
    print(f"Output: {output_dir.absolute()}")
    print("=" * 60)

    # Initialize generator
    generator = DataGenerator(seed=args.seed)

    # Generate all data
    customers = generator.generate_customers(args.customers)
    orders, order_items = generator.generate_orders()
    campaigns = generator.generate_marketing_campaigns(200)
    support_tickets = generator.generate_support_tickets(int(args.customers * 0.3))

    marketing_raw = generator.generate_marketing_customers_raw()

    if not args.skip_sessions:
        sessions = generator.generate_website_sessions(args.customers * 5)
    else:
        sessions = []

    # Save to CSV
    print("\nSaving files...")
    save_csv(customers, output_dir / "customers.csv")
    save_csv(orders, output_dir / "orders.csv")
    save_csv(order_items, output_dir / "order_items.csv")
    save_csv(PRODUCTS, output_dir / "products.csv")
    save_csv(CATEGORIES, output_dir / "categories.csv")
    save_csv(campaigns, output_dir / "marketing_campaigns.csv")
    save_csv(support_tickets, output_dir / "customer_support.csv")
    save_csv(marketing_raw, output_dir / "marketing_customers_raw.csv")

    if sessions:
        save_csv(sessions, output_dir / "website_sessions.csv")

    # Summary
    print("\n" + "=" * 60)
    print("DATA QUALITY ISSUES INTRODUCED (for students to find):")
    print("=" * 60)
    print("Customers:")
    print("  - ~3% duplicate emails (different casing)")
    print("  - ~1.5% missing emails")
    print("  - ~0.8% invalid email format")
    print("  - ~5% missing phone numbers")
    print("  - ~0.5% invalid ages (negative, >150)")
    print("  - ~0.3% future signup dates")
    print("\nOrders:")
    print("  - ~0.8% future order dates")
    print("  - ~0.5% negative totals")
    print("  - ~0.3% missing customer_id")
    print("  - ~0.5% negative quantities in items")
    print("  - October 2024 anomaly: fewer orders + extra cancellations + lower values (~15% revenue drop)")
    print("\nMarketing Raw:")
    print("  - Different column names from customers.csv")
    print("  - ~10% missing values across fields")
    print("  - Whitespace padding on names and cities")
    print("  - Extra junk columns (utm_campaign, notes, etc.)")
    print("\nCampaigns:")
    print("  - ~2% budget overspend")
    print("  - ~1% invalid impressions")
    print("=" * 60)
    print("\nDone! Students can now analyze and clean this data.")


if __name__ == "__main__":
    main()
