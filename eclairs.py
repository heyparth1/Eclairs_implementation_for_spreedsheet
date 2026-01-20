# eclairs_llm_agents.py
# ECLAIR with LLM-based agent detection

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
import json
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# BASE LLM AGENT CLASS
# ============================================================================

class BaseLLMAgent:
    """Base class for LLM-powered agents"""
    
    def __init__(self, schema: Dict[str, Any], llm_client: OpenAI):
        self.schema = schema
        self.llm = llm_client
        self.model = "gpt-4o-mini"  # Fast and cheap
        self.temperature = 0.2  # Slightly increased to avoid being too deterministic
    
    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
        """Centralized LLM call with error handling"""
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ LLM call failed: {str(e)}")
            return json.dumps({"ambiguous": False, "reason": "LLM call failed"})
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse LLM JSON response with fallback"""
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_str = response.split("```json").split("```").strip()[1]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except:
            # Fallback: try to parse YES/NO
            if "YES" in response.upper():
                return {"ambiguous": True, "details": response}
            else:
                return {"ambiguous": False}


# ============================================================================
# 1. SCHEMA GROUNDING AGENT (Same as before)
# ============================================================================

class SchemaGroundingAgent:
    """Extracts and indexes spreadsheet metadata for grounding"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)
        self.schema = self._extract_schema()
        
    def _extract_schema(self) -> Dict:
        if 'Type/SNo' in self.df.columns:
            type_sno_clean = self.df['Type/SNo'].apply(
                lambda x: None if str(x).startswith('C') and len(str(x)) > 4 else x
            )
            type_sno_unique = type_sno_clean.dropna().unique()
            print(f"Type/SNo: {len(self.df['Type/SNo'].unique())} total → {len(type_sno_unique)} product types")
        
        schema = {
            'columns': self.df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'unique_values': {},
            'sample_values': {},
            'numeric_columns': [],
            'categorical_columns': {},
            'high_cardinality_columns': []
        }
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                schema['numeric_columns'].append(col)
                non_null = self.df[col].dropna()
                if len(non_null) > 0:
                    schema['sample_values'][col] = {
                        'samples': non_null.head(3).tolist(),
                        'range': f"{non_null.min():.2f} to {non_null.max():.2f}"
                    }
                continue
            
            if col == 'Type/SNo':
                unique_vals = type_sno_unique
            else:
                unique_vals = self.df[col].dropna().unique()
            
            num_unique = len(unique_vals)
            
            if num_unique < 100 or col in ['Product Type', 'Location', 'Product Description']:
                schema['unique_values'][col] = unique_vals.tolist()
                schema['categorical_columns'][col] = unique_vals.tolist()
            elif num_unique > 1000:
                schema['high_cardinality_columns'].append(col)
                schema['unique_values'][col] = unique_vals[:10].tolist()
            else:
                schema['unique_values'][col] = unique_vals[:20].tolist()
            
            schema['sample_values'][col] = self.df[col].dropna().head(3).tolist()
        
        return schema
    
    def get_grounding_context(self) -> str:
        context = "=== SPREADSHEET SCHEMA ===\n\n"
        context += f"**Dataset**: {len(self.df):,} records, {len(self.schema['columns'])} columns\n\n"
        
        if self.schema.get('categorical_columns'):
            context += "**Categorical Columns**:\n"
            for col, vals in self.schema['categorical_columns'].items():
                context += f"  • {col}: {', '.join(map(str, vals[:10]))}"
                if len(vals) > 10:
                    context += f" (+{len(vals)-10} more)"
                context += "\n"
            context += "\n"
        
        if self.schema['numeric_columns']:
            context += "**Numeric Columns**:\n"
            for col in self.schema['numeric_columns']:
                col_lower = col.lower()
                if 'weight' in col_lower or 'wt' in col_lower:
                    context += f"  • {col} [Weight metric]\n"
                elif 'value' in col_lower or 'price' in col_lower:
                    context += f"  • {col} [Value/Price metric]\n"
                elif 'qty' in col_lower or 'stock' in col_lower:
                    context += f"  • {col} [Quantity metric]\n"
                else:
                    context += f"  • {col}\n"
            context += "\n"
        
        if self.schema.get('high_cardinality_columns'):
            context += "**Identifier Columns**: "
            context += f"{', '.join(self.schema['high_cardinality_columns'])}\n\n"
        
        # Load enhanced domain context if available
        domain_file = os.path.join(os.path.dirname(self.file_path), 'domain_context.md')
        if os.path.exists(domain_file):
            try:
                with open(domain_file, 'r', encoding='utf-8') as f:
                    domain_context = f.read()
                context += "\n=== DOMAIN KNOWLEDGE & CONVENTIONS ===\n"
                context += domain_context + "\n"
            except Exception as e:
                print(f"⚠️ Failed to load domain_context.md: {e}")
        
        return context
    
    def get_agent_output(self) -> Dict[str, Any]:
        return {
            'agent_name': 'Schema Grounding Agent',
            'description': 'Provides spreadsheet structure, columns, and available metrics',
            'context': self.get_grounding_context(),
            'schema': self.schema
        }


# ============================================================================
# 2. LLM-BASED COLUMN AMBIGUITY AGENT
# ============================================================================

class ColumnAmbiguityAgent(BaseLLMAgent):
    """LLM-based column ambiguity detection"""
    
    def __init__(self, schema: Dict[str, Any], llm_client: OpenAI):
        super().__init__(schema, llm_client)
        self.columns = schema['columns']
    
    def detect_ambiguity(self, query: str) -> Dict[str, Any]:
        """Use LLM to detect column ambiguities"""
        
        prompt = f"""You are a STRICT ambiguity detector for spreadsheet queries.

AVAILABLE COLUMNS:
{', '.join(self.columns)}

USER QUERY:
"{query}"

🚨 CRITICAL STRICT RULES - YOU MUST FLAG AS AMBIGUOUS IF:

1. Query mentions "weight" or "wt" → MUST ASK: Which weight column?
   - Wt/Ft (lbs)
   - WT/Pce (lbs)  
   - Total Wt (Tons )

2. Query mentions "value", "cost", or "price" → MUST ASK: Which value metric?
   - Total Value
   - Current Stock

2. Query mentions "value", "cost", or "price" → MUST ASK: Which value metric?
   - Total Value
   - Current Stock

3. Query mentions "type" → MUST ASK: Which type column?
   - Product Type
   - Type/SNo

4. Query mentions "quantity", "qty", or "stock" → MUST ASK: Which quantity?
   - Qty (PCS)
   - Current Stock

🚫 NEGATIVE CONSTRAINTS:
- DO NOT flag "weight" as ambiguous unless the word "weight", "wt", "heaviest", "tons", "lbs" or similar is EXPLICITLY in the query.
- If query is just "List products" or "Show 40s", DO NOT assume it's about weight.

Respond in JSON format:
{{
  "ambiguous": true/false,
  "ambiguous_terms": [
    {{
      "term": "weight",
      "possible_columns": ["Wt/Ft (lbs)", "WT/Pce (lbs)", "Total Wt (Tons )"],
      "reason": "Term 'weight' could refer to multiple weight-related columns"
    }}
  ]
}}

If not ambiguous, return: {{"ambiguous": false, "ambiguous_terms": []}}
"""
        
        response = self._call_llm(prompt, max_tokens=400)
        result = self._parse_json_response(response)
        
        return {
            'ambiguous': result.get('ambiguous', False),
            'ambiguities': result.get('ambiguous_terms', []),
            'raw_response': response
        }
    
    def get_agent_output(self, query: str) -> Dict[str, Any]:
        result = self.detect_ambiguity(query)
        output = {
            'agent_name': 'Column Ambiguity Agent (LLM)',
            'description': 'Uses LLM to identify ambiguous column references',
            'ambiguous': result['ambiguous'],
            'details': None
        }
        
        if result['ambiguous'] and result.get('ambiguities'):
            details = "Detected ambiguous column references:\n"
            for amb in result['ambiguities']:
                term = amb.get('term', 'unknown')
                cols = amb.get('possible_columns', [])
                reason = amb.get('reason', '')
                details += f"  • Term '{term}' could refer to: {', '.join(cols)}\n"
                if reason:
                    details += f"    Reason: {reason}\n"
            output['details'] = details
        
        return output


# ============================================================================
# 3. LLM-BASED VALUE AMBIGUITY AGENT
# ============================================================================

class ValueAmbiguityAgent(BaseLLMAgent):
    """LLM-based value/entity ambiguity detection"""
    
    def __init__(self, df: pd.DataFrame, schema: Dict[str, Any], llm_client: OpenAI):
        super().__init__(schema, llm_client)
        self.df = df
        self.categorical_columns = schema.get('categorical_columns', {})
    
    def detect_ambiguity(self, query: str) -> Dict[str, Any]:
        """Use LLM to detect value ambiguities"""
        
        # Build categorical values context
        cat_context = ""
        for col, values in self.categorical_columns.items():
            cat_context += f"- {col}: {', '.join(map(str, values[:15]))}\n"
        
        prompt = f"""You are a STRICT value ambiguity detector for spreadsheet queries.

CATEGORICAL COLUMNS AND VALUES:
{cat_context}

DOMAIN CONTEXT (Naming Conventions & Codes):
{self.schema.get('domain_context_snippet', 'See full schema context for details')}

USER QUERY:
"{query}"

🚨 YOUR JOB: Detect Country, Entity, and NUMERIC/DIMENSION ambiguity.

1. Country/Entity Ambiguity:
   - "USA", "Canada", "Korea" → These are **ORIGIN COUNTRIES**.
   - Rule: **Origins are stored in 'Product Description'**.
   - STATUS: **CLEAR** (Do not flag as ambiguous. SQL generator handles this.)
   - "North", "South" → Vague location? (AMBIGUOUS)

2. NUMERIC / DIMENSION AMBIGUITY (CRITICAL):

2. NUMERIC / DIMENSION AMBIGUITY (CRITICAL):
   - "40s", "20s", "60s", "8x8" → These are ambiguous numbers.
   - Refer to **Measurement Definitions** in Context.
   - Could be:
     - **Length** (in Feet) - e.g., 40ft
     - **OD / Width** (in Inches) - e.g., 40" OD
     - **Wall Thickness** (in Inches)
     - **Grade** (e.g., Grade 40)
   - 🚫 NEVER suggest "Schedule", "Height", or "Class". They do not exist.

   EXAMPLE: "List the 40s"
   → AMBIGUOUS: "Do you mean 40ft Length, 40 inch OD/Width, or Grade 40?"

3. DO NOT FLAG:
   - Column names (COLUMN agent job). **ESPECIALLY IGNORE**:
     - "weight", "wt", "value", "price", "cost", "stock", "quantity".
     - These are updated by the COLUMN agent. DO NOT FLAG THEM AS AMBIGUOUS VALUES.
   - Explicit filters ("Length 40", "Grade 50")

Respond in JSON format:
{{
  "ambiguous": true/false,
  "ambiguous_entities": [
    {{
      "entity": "40s",
      "type": "numeric_ambiguity",
      "issue": "Could be Length, OD/Width, or Grade",
      "possible_meanings": ["40ft Length", "40 inch OD", "Grade 40"]
    }}
  ]
}}

If not ambiguous, return: {{"ambiguous": false, "ambiguous_entities": []}}
"""
        
        response = self._call_llm(prompt, max_tokens=400)
        result = self._parse_json_response(response)
        
        return {
            'ambiguous': result.get('ambiguous', False),
            'ambiguities': result.get('ambiguous_entities', []),
            'raw_response': response
        }
    
    def get_agent_output(self, query: str) -> Dict[str, Any]:
        result = self.detect_ambiguity(query)
        output = {
            'agent_name': 'Value Ambiguity Agent (LLM)',
            'description': 'Uses LLM to identify ambiguous entity references',
            'ambiguous': result['ambiguous'],
            'details': None
        }
        
        if result['ambiguous'] and result.get('ambiguities'):
            details = "Detected ambiguous entity references:\n"
            for amb in result['ambiguities']:
                entity = amb.get('entity', 'unknown')
                issue = amb.get('issue', '')
                contexts = amb.get('contexts', [])
                details += f"  • '{entity}': {issue}\n"
                if contexts:
                    details += f"    Contexts: {', '.join(map(str, contexts[:5]))}\n"
            output['details'] = details
        
        return output


# ============================================================================
# 4. LLM-BASED AGGREGATION AMBIGUITY AGENT
# ============================================================================

class AggregationAmbiguityAgent(BaseLLMAgent):
    """LLM-based aggregation ambiguity detection"""
    
    def __init__(self, schema: Dict[str, Any], llm_client: OpenAI):
        super().__init__(schema, llm_client)
        self.numeric_columns = schema['numeric_columns']
        self.categorical_columns = list(schema.get('categorical_columns', {}).keys())
    
    def detect_ambiguity(self, query: str) -> Dict[str, Any]:
        """Use LLM to detect aggregation ambiguities"""
        
        prompt = f"""You are a STRICT aggregation ambiguity detector.

SPREADSHEET SCHEMA:
- Numeric columns: {', '.join(self.numeric_columns)}
- Categorical columns: {', '.join(self.categorical_columns)}

USER QUERY:
"{query}"

🚨 STRICT RULES - FLAG AS AMBIGUOUS IF QUERY:

1. Uses aggregation words (total, sum, average) BUT:
   - Does NOT specify exact metric column name
   - Does NOT specify grouping ("by Location", "by Product Type", or "overall")

2. SPECIAL RULE FOR "COUNT":
   - "Count" does NOT require a metric column (it implies counting rows/size).
   - "Count the beams" → CLEAR (if 'beams' is a filter).
   - "Count the pipes at each location" → CLEAR (Grouping: Location).
   - ONLY flag "Count" as ambiguous if the grouping is unclear or conflict (e.g., "count by type and location or something else?").

3. "List" or "Show" are NOT aggregations (they are Filters/Row Selection):
   - "List the 40s" → NOT AMBIGUOUS for Aggregation. It means "Select Rows".
   - "List the 40s by Width" → THIS IS SORTING, NOT AGGREGATION. Result: false.

4. Be EXTREMELY STRICT with METRICS:
   ❌ "total weight" → AMBIGUOUS (which weight? group by what?)
   ✅ "Total Wt (Tons )" → CLEAR
   ✅ "List the 40s" → CLEAR (Row Selection only)

5. If query is vague or uses generic terms like "average", FLAG AS AMBIGUOUS

6. 🚫 NEGATIVE CONSTRAINTS (CRITICAL):
   - NEVER suggest metrics that are NOT in the "Numeric columns" list above.
   - "Length", "Width", "Height", "Schedule" are NOT columns (except OD=Width). DO NOT suggest aggregating them.
   - We CANNOT calculate "average length" or "total length".
   - If user asks for "average length", FAIL GRACEFULLY or ask if they mean "Avg Wt/Ft".

Respond in JSON format:
{{
  "ambiguous": true/false,
  "issues": [
    {{
      "type": "missing_grouping",
      "reason": "Query requests total but doesn't specify grouping dimension",
      "suggestions": ["Product Type", "Location"]
    }},
    {{
      "type": "unclear_metric",
      "reason": "Term 'weight' could refer to multiple weight columns",
      "possible_metrics": ["Wt/Ft (lbs)", "Total Wt (Tons )"]
    }}
  ]
}}

If not ambiguous, return: {{"ambiguous": false, "issues": []}}
"""
        
        response = self._call_llm(prompt, max_tokens=400)
        result = self._parse_json_response(response)
        
        return {
            'ambiguous': result.get('ambiguous', False),
            'ambiguities': result.get('issues', []),
            'raw_response': response
        }
    
    def get_agent_output(self, query: str) -> Dict[str, Any]:
        result = self.detect_ambiguity(query)
        output = {
            'agent_name': 'Aggregation Ambiguity Agent (LLM)',
            'description': 'Uses LLM to detect unclear aggregation intent',
            'ambiguous': result['ambiguous'],
            'details': None
        }
        
        if result['ambiguous'] and result.get('ambiguities'):
            details = "Detected aggregation ambiguities:\n"
            for amb in result['ambiguities']:
                amb_type = amb.get('type', 'unknown')
                reason = amb.get('reason', '')
                suggestions = amb.get('suggestions', [])
                details += f"  • {reason}\n"
                if suggestions:
                    details += f"    Suggestions: {', '.join(suggestions)}\n"
            output['details'] = details
        
        return output


# ============================================================================
# 5. LLM-BASED FILTER AMBIGUITY AGENT
# ============================================================================

class TemporalFilterAmbiguityAgent(BaseLLMAgent):
    """LLM-based filter/temporal ambiguity detection"""
    
    def __init__(self, schema: Dict[str, Any], llm_client: OpenAI):
        super().__init__(schema, llm_client)
        self.categorical_columns = schema.get('categorical_columns', {})
    
    def detect_ambiguity(self, query: str) -> Dict[str, Any]:
        """Use LLM to detect filter/temporal ambiguities"""
        
        # Build categorical values context
        cat_context = ""
        for col, values in self.categorical_columns.items():
            # Take only first 30 values to avoid token limit, but enough for context
            cat_context += f"- {col} Options: {', '.join(map(str, values[:30]))}\n"
        
        prompt = f"""You are a STRICT filter ambiguity detector for spreadsheet queries.

AVAILABLE FILTERS AND DATA VALUES:
{cat_context}

USER QUERY:
"{query}"

🚨 STRICT RULES for detecting ambiguity:

1. **CRITICAL: "Generic Location Terms" Check (HIGHEST PRIORITY)**:
   - If query mentions "yard", "warehouse", "site", or "location" (roughly matches) BUT *DOES NOT* mention a specific code (like PPBC, PPMTL, etc.), you **MUST FLAG AS AMBIGUOUS**.
     - Query: "How much weight is in the yard?"
     - Issue: "User mentioned 'yard' but didn't specify WHICH yard."
     - Action: Return ambiguous=true, suggests: ["PPBC", "PPMTL", "PPCAL", "etc..."]

2. CHECK IF VALUE ALREADY EXISTS:
   - If query mentions a value that matches one of the "Options" above (case-insensitive), DO NOT ask for that filter.
   - Example: If "Product Type" options include "BEAMS", it IS specified.

3. ONLY SUGGEST VALID OPTIONS:
   - NEVER hallucinate options. Only suggest values listed in "AVAILABLE FILTERS" above.
   - If you ask for a "Product Type", you MUST list actual values from the data (e.g., "BEAMS", "PIPE", "SHEET", "COILS").
   - NEVER suggest "40s", "OD", "Length", or query terms as Product Types.

4. FLAG AS AMBIGUOUS ONLY IF:
   - Query is broad (e.g., "Show inventory") AND no specific value from "AVAILABLE FILTERS" is mentioned.
   - A critical filter (Location, Product Type) is completely missing AND not implied by the values.

Examples:
✅ "List beams" → Product Type is CLEAR (matches 'BEAMS'). Check if Location is missing.
✅ "Show BEAMS at PPMTL" → ALL CLEAR (matches 'BEAMS' and 'PPMTL').
❌ "Show inventory" → AMBIGUOUS (No specific values mentioned).
❌ "Weight in the yard" → AMBIGUOUS (Which yard? PPBC? PPMTL?).
❌ "Show columns" → AMBIGUOUS (If 'columns' is NOT in Options, then it's invalid/unclear. If it IS in Options, it's clear).

Respond in JSON format:
{{
  "ambiguous": true/false,
  "issues": [
    {{
      "type": "missing_filter",
      "reason": "Query is too broad...",
      "suggestions": ["Product Type", "Location"]
    }}
  ]
}}

If not ambiguous, or if values are present, return: {{"ambiguous": false, "issues": []}}
"""
        
        response = self._call_llm(prompt, max_tokens=300)
        result = self._parse_json_response(response)
        
        return {
            'ambiguous': result.get('ambiguous', False),
            'ambiguities': result.get('issues', []),
            'raw_response': response
        }
    
    def get_agent_output(self, query: str) -> Dict[str, Any]:
        result = self.detect_ambiguity(query)
        output = {
            'agent_name': 'Filter Ambiguity Agent (LLM)',
            'description': 'Uses LLM to detect missing filter criteria',
            'ambiguous': result['ambiguous'],
            'details': None
        }
        
        if result['ambiguous'] and result.get('ambiguities'):
            details = "Detected filter/scope ambiguities:\n"
            for amb in result['ambiguities']:
                reason = amb.get('reason', '')
                suggestions = amb.get('suggestions', [])
                details += f"  • {reason}\n"
                if suggestions:
                    details += f"    Consider filtering by: {', '.join(suggestions)}\n"
            output['details'] = details
        
        return output


# ============================================================================
# 6. OPTIMIZED ECLAIR SYSTEM WITH PARALLEL EXECUTION
# ============================================================================

class ECLAIRSpreadsheetSystem:
    """ECLAIR system with LLM-based agents and parallel execution"""
    
    def __init__(self, file_path: str, api_key: Optional[str] = None):
        print(f"🔧 Initializing ECLAIR System with LLM Agents...")
        
        # Initialize LLM client
        self.llm_client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        # Load data
        self.df = pd.read_excel(file_path)
        
        # FIX: Ensure 'Type/SNo' is string to prevent PyArrow serialization errors with mixed types
        if 'Type/SNo' in self.df.columns:
            self.df['Type/SNo'] = self.df['Type/SNo'].astype(str)
        
        # Initialize schema grounding
        self.schema_agent = SchemaGroundingAgent(file_path)
        schema = self.schema_agent.schema
        
        # Initialize LLM-based agents
        self.agents = {
            'schema': self.schema_agent,
            'column': ColumnAmbiguityAgent(schema, self.llm_client),
            'value': ValueAmbiguityAgent(self.df, schema, self.llm_client),
            'aggregation': AggregationAmbiguityAgent(schema, self.llm_client),
            'filter': TemporalFilterAmbiguityAgent(schema, self.llm_client)
        }
        
        print(f"✅ ECLAIR System initialized with {len(self.df):,} records")
        print(f"✅ Loaded {len(self.agents)} agents (4 LLM-based)\n")
    
    def build_unified_prompt(self, query: str, agent_outputs: Dict) -> str:
        """Build unified ECLAIR prompt"""
        grounding_context = self.agents['schema'].get_grounding_context()
        
        prompt = f"""You are an AI assistant for spreadsheet Q&A that generates clarification questions.

## SPREADSHEET CONTEXT
{grounding_context}

## USER QUERY
"{query}"

## AGENT ANALYSIS
"""
        
        agent_number = 1
        for name, output in agent_outputs.items():
            prompt += f"\n### Agent {agent_number}: {output['agent_name']}\n"
            prompt += f"**Function**: {output['description']}\n"
            prompt += f"**Ambiguity Detected**: {'YES' if output['ambiguous'] else 'NO'}\n"
            if output['ambiguous'] and output['details']:
                prompt += f"**Details**:\n{output['details']}\n"
            agent_number += 1
        
        prompt += """
## YOUR TASK

Based on agent analysis:
1. Determine if query is ambiguous (YES/NO)
2. If YES: Generate ONE clarification question.

## PRIORITIZATION RULES (CRITICAL):
1. **FILTER / SCOPE (HIGHEST PRIORITY)**:
   - If User Ambiguity Agent detected "missing location" (e.g., "Which yard?") or "missing product type", **ASK THIS FIRST**.
   - We cannot answer *what* (metrics) if we don't know *where* (location) or *which items* (product type).
2. **COLUMN / METRIC**:
   - Ask about "Total Wt vs Wt/Ft" only if Location is clear.
3. **AGGREGATION**:
   - Ask about grouping last.

## OUTPUT FORMAT

AMBIGUOUS: [YES or NO]
CLARIFICATION: [Your question with options, or "None"]

Example: "Which yard are you referring to? Options: PPBC, PPMTL, etc."
"""
        
        return prompt
    
    
    def process_query_parallel(self, query: str, excluded_agents: set = None) -> Dict[str, Any]:
        """Process query with parallel agent execution, optionally skipping excluded agents"""
        if excluded_agents is None:
            excluded_agents = set()
        
        print(f"🔍 Processing query: '{query}'")
        if excluded_agents:
            print(f"   Skipping cleared agents: {', '.join([a.upper() for a in excluded_agents])}")
        print("   Running agents in parallel...")
        
        agent_outputs = {}
        
        # Run LLM agents in parallel (skip excluded ones)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_agent = {}
            
            for name, agent in self.agents.items():
                if name == 'schema':
                    continue
                if name in excluded_agents:
                    # Mark as already cleared
                    agent_outputs[name] = {
                        'agent_name': f'{name.title()} Agent (LLM)',
                        'description': 'Already cleared in previous interaction',
                        'ambiguous': False,
                        'details': '✅ Cleared'
                    }
                    print(f"   {name.upper():15s} ⏭️  SKIPPED (already cleared)")
                    continue
                    
                future = executor.submit(agent.get_agent_output, query)
                future_to_agent[future] = name
            
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    output = future.result(timeout=10)
                    agent_outputs[agent_name] = output
                    status = "🔴 AMBIGUOUS" if output['ambiguous'] else "🟢 CLEAR"
                    print(f"   {agent_name.upper():15s} {status}")
                except Exception as e:
                    print(f"   ⚠️ {agent_name.upper()} failed: {str(e)}")
                    agent_outputs[agent_name] = {
                        'agent_name': f'{agent_name} Agent',
                        'description': 'Failed to execute',
                        'ambiguous': False,
                        'details': None
                    }
        
        # Build unified prompt
        print("\n   Building unified prompt...")
        prompt = self.build_unified_prompt(query, agent_outputs)
        
        # Final LLM call
        print("   Calling final LLM for clarification...")
        final_response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        llm_response = final_response.choices[0].message.content
        
        # Parse response
        ambiguous_match = re.search(r'AMBIGUOUS:\s*(YES|NO)', llm_response, re.IGNORECASE)
        is_ambiguous = ambiguous_match.group(1).upper() == 'YES' if ambiguous_match else False
        
        clarification_match = re.search(r'CLARIFICATION:\s*(.+?)(?:\n\n|\Z)', llm_response, re.DOTALL | re.IGNORECASE)
        clarification = clarification_match.group(1).strip() if clarification_match else None
        
        if clarification and clarification.lower() == 'none':
            clarification = None
        
        print("   ✅ Processing complete!\n")
        
        return {
            'query': query,
            'ambiguous': is_ambiguous,
            'clarification': clarification,
            'agent_outputs': agent_outputs,
            'prompt': prompt,
            'raw_response': llm_response
        }


# ============================================================================
# 7. MAIN TESTING CODE
# ============================================================================

if __name__ == "__main__":
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("   Set it with: $env:OPENAI_API_KEY='your-key-here'  (Windows)")
        print("   Or: export OPENAI_API_KEY='your-key-here'  (Linux/Mac)")
        exit(1)
    
    # Initialize ECLAIR
    eclair = ECLAIRSpreadsheetSystem('pipeandpillings.xls', api_key)
    
    # Test queries
    test_queries = [
        "What is the total weight?",
        "Show me all BEAMS",
        "List products at PPBC with weight over 100 lbs"
    ]
    
    print("\n" + "="*80)
    print("TESTING ECLAIR WITH LLM AGENTS")
    print("="*80 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'═'*80}")
        print(f"TEST {i}: {query}")
        print('═'*80 + "\n")
        
        result = eclair.process_query_parallel(query)
        
        print("="*80)
        print("RESULT:")
        print("="*80)
        print(f"Ambiguous: {result['ambiguous']}")
        
        if result.get('clarification'):
            print(f"\nClarification Question:")
            print(f"  {result['clarification']}")
        else:
            print("\n✓ No clarification needed - query is clear!")
        
        print("\n")
