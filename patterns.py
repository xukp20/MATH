"""
    Examples and Prompts
"""

EXAMPLES = [{
    "problem": "Find the domain of the expression $\\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}",
    "solution": """The expressions inside each square root must be non-negative. Therefore,
$x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator
cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of
the expression is $\\boxed{[2,5)}$.""",
    "description": "",
    "answer": "[2,5)",
}]

INSTRUCTION_W_DESC = """
### Instruction:
You are a top-notch mathematician. Your task is to provide the solution of the given problem, with the final answer boxed by \\boxed{}.
Some tasks have a description of the picture to help you understand the problem if "Description" is not empty.
----

"""

INSTRUCTION_WO_DESC = """
### Instruction:
You are a top-notch mathematician. Your task is to provide the solution of the given problem, with the final answer boxed by \\boxed{}.
----

"""

PATTERN_W_DESC = INSTRUCTION_W_DESC + """
{{~! display the few-shot examples ~}}
{{~#each examples}}
### Input:
Problem: {{this.problem}}
Description: {{this.description}}

### Response:
Solution: {{this.solution}}
Answer: {{this.answer}}
---
{{~/each}}

{{~! place the real question at the end }}
### Input:
Problem: {{problem}}
Description: {{description}}

### Response:
Solution: {{gen "solution" temperature=0. max_tokens=400 stop='Answer'}}
Answer: {{gen "answer" temperature=0. max_tokens=50 stop='\n'}}
"""

PATTERN_WO_DESC = INSTRUCTION_WO_DESC + """
{{~! display the few-shot examples ~}}
{{~#each examples}}
### Input:
Problem: {{this.problem}}

### Response:
Solution: {{this.solution}}
Answer: {{this.answer}}
---
{{~/each}}

{{~! place the real question at the end }}
### Input:
Problem: {{problem}}

### Response:
Solution: {{gen "solution" temperature=0. max_tokens=400 stop='Answer'}}
Answer: {{gen "answer" temperature=0. max_tokens=50 stop='\n'}}
"""