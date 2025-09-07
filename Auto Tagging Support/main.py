from utils import safe_classification

# Sample
tickets = [
    "Very good",
    "My app keeps crashing when I open it",
    "I want a refund for my order",
    "Password reset is not working",
    "Where is my package?"
]

for remark in tickets:
    print("Ticket Remark:", remark)

    z_result = safe_classification(remark, few_shot=False)  
    f_result = safe_classification(remark, few_shot=True)   

    print("Zero-Shot Result:", z_result)
    print("Few-Shot Result:", f_result)
    print("-" * 50)
