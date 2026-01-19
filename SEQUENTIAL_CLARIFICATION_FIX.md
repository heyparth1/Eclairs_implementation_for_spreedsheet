# Sequential Clarification Fix

## Problem
The system was asking about MULTIPLE agents in ONE clarification question:

```
❌ "Would you like to list by Product Type or Location? 
   Also, which metric: Current Stock, Total Wt, or Total Value?"
```

This defeats the purpose of sequential clarification.

## Root Cause
The `build_unified_prompt()` in `eclairs.py` combines ALL ambiguous agents into one prompt, so the LLM generates a combined clarification addressing multiple issues.

## Solution
Generate clarification for **ONE agent at a time** using new function:

### `generate_single_agent_clarification()`
```python
def generate_single_agent_clarification(agent_name, agent_output, query, client):
    # Generates clarification for ONLY this specific agent
    # Focuses on ONE concern
    # Returns single question
```

## Now Works Like:

**Round 1:**
```
🔍 Checking COLUMN agent...
Clarification: "Which weight metric? Wt/Ft (lbs), WT/Pce (lbs), or Total Wt (Tons)?"
```

**User answers:** "Total Wt"
✅ COLUMN cleared

**Round 2:**
```
🔍 Checking AGGREGATION agent...
Clarification: "How to group? By Location, Product Type, or overall total?"
```

**User answers:** "by location"
✅ AGGREGATION cleared

**All clear** → Generate SQL!

## Key Change
Instead of using `summary.get('clarification')` (unified for all agents), we now call `generate_single_agent_clarification()` for just the current agent.
