
import os
import requests
from openai import OpenAI
from config import (
    OPENAI_API_KEY, GROQ_API_KEY, OPENAI_MODEL, GROQ_MODEL, GROQ_URL,
    CATEGORIES, SUBCATEGORIES
)

# Init OpenAI 
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# =====================
# Prompt 
# =====================
def build_prompt(remark, few_shot=False):
    examples = ""
    if few_shot:
        examples = """
        Example 1:
        Ticket: "I want to return my phone, it’s broken."
        Category: Returns
        Subcategory: Exchange / Replacement

        Example 2:
        Ticket: "I was charged twice for my order."
        Category: Payments
        Subcategory: Double Charge
        """

    return f"""
You are a support ticket classifier.
Categories: {CATEGORIES}
Subcategories: {SUBCATEGORIES}

{examples}

Ticket: "{remark}"
Return JSON with: Category, Subcategory, and Top_3 most probable categories.
"""

# =====================
# OpenAI Classification
# =====================
def openai_classification(remark, few_shot=False):
    prompt = build_prompt(remark, few_shot)
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# =====================
# Groq Classification
# =====================
def groq_classification(remark, few_shot=False):
    prompt = build_prompt(remark, few_shot)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# =====================
# Mock Classification (offline testing)
# =====================
def mock_classification(remark, few_shot=False):
    return {
        "Category": "Returns",
        "Subcategory": "Exchange / Replacement",
        "Top_3": ["Returns", "Payments", "General Inquiry"]
    }

# =====================
# Safe Classification (fallback chain)
# =====================
def safe_classification(remark, few_shot=False):
    try:
        return openai_classification(remark, few_shot=few_shot)
    except Exception as e1:
        print("OpenAI failed, trying Groq...", e1)
        try:
            return groq_classification(remark, few_shot=few_shot)
        except Exception as e2:
            print("Groq also failed, using Mock...", e2)
            return mock_classification(remark, few_shot=few_shot)
