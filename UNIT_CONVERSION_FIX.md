# Unit Preservation and Conversion Implementation

## Problem
User asked: "total weight of BEAMS in kgs"
Reformulator changed it to: "total weight of BEAMS in Tons" ❌

This violated user intent because the reformulator "helpfully" changed units to match available columns.

## Solution: Two-Part Fix

### Part 1: Preserve Units in Reformulation
Updated `reformulate_query_with_answer()` prompt:

**Added Critical Guideline:**
```
PRESERVE ALL UNITS mentioned by user (kgs, tons, lbs, etc.) - DO NOT CHANGE THEM
```

**Example:**
```
Original: "show weight in kgs"
Answer: "Total Wt"
Reformulated: "show Total Wt in kgs"  ← PRESERVED "in kgs"
```

### Part 2: Unit Conversion in SQL Generator
Updated `generate_sql_query()` to handle conversions:

**Conversion Rules:**
```python
# Spreadsheet has:
- Total Wt (Tons)  → in TONS
- Wt/Ft (lbs)      → in POUNDS per foot
- WT/Pce (lbs)     → in POUNDS per piece

# Conversions:
- Tons to kgs: × 1000
- Tons to lbs: × 2204.62
- lbs to kgs: × 0.453592
- lbs to tons: ÷ 2204.62
- kgs to tons: ÷ 1000
- kgs to lbs: ÷ 0.453592
```

**Generated Code Example:**
```python
# User: "total weight in kgs"
result = df['Total Wt (Tons )'].sum() * 1000  # Auto-converts tons to kgs
```

## Flow Now

**User Query:** "total weight of BEAMS in kgs"

**Reformulation:** "total weight of BEAMS in kgs"  
✅ Units preserved

**SQL Generated:**
```python
result = df[df['Product Type'] == 'BEAMS']['Total Wt (Tons )'].sum() * 1000
```
✅ Converts tons to kgs

**Result:** Weight in kilograms as requested!

## Benefits
1. ✅ Respects user's unit preference
2. ✅ Automatic conversion between tons/kgs/lbs
3. ✅ No manual unit tracking needed
4. ✅ Works for any unit combination
