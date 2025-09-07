# config.py

# =====================
# API 
# =====================
OPENAI_API_KEY = "your-openai-api-key"
GROQ_API_KEY = "your-groq-api-key"

# =====================
# MODELS
# =====================
OPENAI_MODEL = "gpt-4o-mini"   
GROQ_MODEL = "llama3-70b-8192"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# =====================
# TAGGING SCHEMA
# =====================
CATEGORIES = [
    "Returns", "Payments", "Technical Issues", "Account", "General Inquiry"
]

SUBCATEGORIES = {
    "Returns": ["Exchange / Replacement", "Refunds", "Damaged Item"],
    "Payments": ["Failed Transaction", "Double Charge", "Billing Issue"],
    "Technical Issues": ["App Crash", "Login Problem", "Slow Performance"],
    "Account": ["Password Reset", "Update Info", "Security Concern"],
    "General Inquiry": ["Product Info", "Delivery Status", "Other"]
}
