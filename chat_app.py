"""
ECLAIR Streamlit Chat Application
Interactive query clarification system with SQL generation
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
import os
import json
from openai import OpenAI
from eclairs import ECLAIRSpreadsheetSystem
import re

# Page configuration
st.set_page_config(
    page_title="ECLAIR Query Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'eclair' not in st.session_state:
    st.session_state.eclair = None

if 'expanded_query' not in st.session_state:
    st.session_state.expanded_query = ""

if 'clarification_count' not in st.session_state:
    st.session_state.clarification_count = 0

# Track which agents have been cleared
if 'cleared_agents' not in st.session_state:
    st.session_state.cleared_agents = set()

# Track current ambiguous agent
if 'current_ambiguous_agent' not in st.session_state:
    st.session_state.current_ambiguous_agent = None

# Track last clarification question for reformulation
if 'last_clarification' not in st.session_state:
    st.session_state.last_clarification = None

# Track query history for cross-query memory
if 'query_history' not in st.session_state:
    st.session_state.query_history = []


def initialize_eclair():
    """Initialize ECLAIR system"""
    try:
        eclair = ECLAIRSpreadsheetSystem('pipeandpillings.xls')
        return eclair
    except Exception as e:
        st.error(f"Failed to initialize ECLAIR system: {str(e)}")
        return None


def get_openai_client():
    """Get OpenAI client"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found. Please set it in your environment.")
        st.stop()
    return OpenAI(api_key=api_key)


def llm_generate(prompt: str, client: OpenAI) -> str:
    """Generate response from OpenAI"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content


def parse_llm_clarification(response: str) -> Dict[str, Any]:
    """Parse LLM clarification response"""
    try:
        ambiguous_match = re.search(r'AMBIGUOUS:\s*(YES|NO)', response, re.IGNORECASE)
        is_ambiguous = ambiguous_match.group(1).upper() == 'YES' if ambiguous_match else False
        
        clarification_match = re.search(r'CLARIFICATION:\s*(.+?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        clarification = clarification_match.group(1).strip() if clarification_match else None
        
        if clarification and clarification.lower() == 'none':
            clarification = None
        
        return {
            'ambiguous': is_ambiguous,
            'clarification': clarification
        }
    except:
        return {'ambiguous': False, 'clarification': None}


def generate_sql_query(query: str, eclair, client: OpenAI) -> str:
    """Generate pandas query code from natural language using schema context"""
    schema_context = eclair.schema_agent.get_grounding_context()
    
    prompt = f"""You are a pandas expert that generates Python code for spreadsheet queries.

SCHEMA:
{schema_context}

USER QUERY:
"{query}"

CRITICAL FILTER RULES:
1. If query mentions specific locations (PPBC, PPCAL, PPMTL, etc.) → MUST filter by Location column
2. If query mentions specific product types (BEAMS, COILS, PIPE, SHEET) → MUST filter by Product Type
3. If query mentions **Countries** (Korea, USA, Canada, etc.) → MUST filter 'Product Description' using str.contains()
   - Example: `df[df['Product Description'].str.contains('Korea', case=False, na=False)]`
4. If query mentions **Numbers/Dimensions** (e.g., "40s", "20s", "8x8"):
   - MUST APPLY AS A FILTER to the relevant column (Length, OD, or Wall).
   - **OD**: Use `df['OD']` column.
   - **Length** (Beams/Pipes): use `df['Product ID'].str[11:16]`.
     - "List the 40s" (clarified as Length) → 
       `df[pd.to_numeric(df['Product ID'].str[11:16], errors='coerce') == 40]`
   - DO NOT just sort by the column. YOU MUST FILTER.
5. Extract ALL mentioned filters and apply them BEFORE grouping/aggregating

ROW SELECTION vs AGGREGATION (CRITICAL):
- "List", "Show", "Find", "Get" (without words like "Count", "Sum", "Total") → **ROW SELECTION**
  - Do NOT use `groupby()` or `.size()`.
  - Return the filtered DataFrame (or select key columns: Product Type, Location, Product Description, OD, Length, Current Stock).
  - Example: `result = df[df['OD'] == 40][['Product Type', 'Location', 'Product Description', 'OD', 'Current Stock']]`
  - **SORTING**: If user asks to "sort by X" or implies it ("heavy", "largest"), ADD `.sort_values()` to the result.
    - "List the 40s" (user implies sort) -> Filter OD=40, then optionally sort by OD.
- "Count", "Sum", "Total", "Average" → **AGGREGATION**
  - Use `groupby().sum()` or `groupby().size()`.

QUALITATIVE SORTING RULES:
1. "Heavy", "Heaviest", "Most Weight" → `df.sort_values(by='[Weight_Column]', ascending=False)`
2. "Light", "Lightest" → `df.sort_values(by='[Weight_Column]', ascending=True)`
3. "Expensive", "Highest Value" → `df.sort_values(by='Total Value', ascending=False)`
4. "Largest", "Biggest" (if referring to OD/Width) → `df.sort_values(by='OD', ascending=False)`
5. "Longest" → `df.sort_values(by='Length', ascending=False)` (Requires extracting Length first if not a column, but '40s' filter handles this usually).




FILTER EXAMPLES:
Query: "show Total Wt for PPBC" 
Code: result = df[df['Location'] == 'PPBC']['Total Wt (Tons )'].sum()

Query: "show BEAMS at PPMTL grouped by Product Type"
Code: result = df[(df['Location'] == 'PPMTL') & (df['Product Type'] == 'BEAMS')].groupby('Product Type')['Total Wt (Tons )'].sum()

Query: "Total Wt grouped by Product Type and Location for PPBC"
Code: result = df[df['Location'] == 'PPBC'].groupby(['Product Type', 'Location'])['Total Wt (Tons )'].sum()

UNIT CONVERSION RULES (CRITICAL):
The spreadsheet has weight columns in specific units:
- 'Total Wt (Tons )'  → in TONS (Multiply by 2000 to get LBS)
- 'Wt/Ft (lbs)'       → in LBS per foot
- 'WT/Pce (lbs)'      → in LBS per piece

1. IF USER DOES NOT SPECIFY UNIT: Use the column's default unit.
   - "Total weight" -> `df['Total Wt (Tons )'].sum()` (Result is in Tons)

2. IF USER SPECIFIES 'LBS' or 'POUNDS':
   - For 'Total Wt': MUST CONVERT → `df['Total Wt (Tons )'].sum() * 2000`
   - For 'Wt/Ft': No conversion needed.

3. IF USER SPECIFIES 'TONS':
   - For 'Total Wt': No conversion needed.
   - For 'Wt/Ft': `df['Wt/Ft (lbs)'].mean() / 2000`

4. LABEL YOUR ANSWER: When explaining the result, YOU MUST STATE THE UNIT.
   - "The total weight is 50,000 lbs" (if converted).
   - "The total weight is 25 tons".

If user asks for different units, CONVERT:
- Tons to kgs: multiply by 1000
- Tons to lbs: multiply by 2204.62
- lbs to kgs: multiply by 0.453592

INSTRUCTIONS:
1. FIRST: Extract all filter criteria from query (locations, product types, etc.)
2. Apply filters using df[...] before any grouping
3. Then apply grouping/aggregation
4. Finally apply unit conversion if needed
5. Use df as the DataFrame variable
6. Store result in 'result' variable
7. Return ONLY valid pandas code, no explanations

Generate the pandas code:
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    code = response.choices[0].message.content.strip()
    
    # Clean up code (remove markdown fences if present)
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    return code


def execute_query(code: str, df: pd.DataFrame) -> Any:
    """Execute pandas query code"""
    try:
        local_vars = {'df': df, 'pd': pd}
        exec(code, {}, local_vars)
        if 'result' in local_vars:
            return local_vars['result']
        return "Query executed but no result variable found"
    except Exception as e:
        return f"Error executing query: {str(e)}"


def format_results(query: str, results: Any, client: OpenAI) -> str:
    """Format query results in human-readable way"""
    prompt = f"""You are a helpful assistant that explains data analysis results.

User asked: "{query}"

Query results:
{str(results)}

CRITICAL INSTRUCTIONS:
- Treat ALL codes and abbreviations as LITERAL identifiers from the dataset
- DO NOT interpret what abbreviations might mean (e.g., "PPMTL" is just a location code, not an acronym)
- DO NOT expand location codes or product codes into phrases
- Location names like PPMTL, PPBC, PPCAL are just identifiers - treat them as-is
- Product types like BEAMS, COILS, PIPE are literal category names
- Focus on explaining the NUMBERS and PATTERNS in the data, not interpreting the names

Provide a clear, concise summary of these results in natural language.
If the results show aggregated data, explain what the numbers mean.
Keep it brief and user-friendly.

Your response:
"""
    
    return llm_generate(prompt, client)


def reformulate_query_with_answer(original_query: str, clarification_question: str, user_answer: str, client: OpenAI) -> str:
    """Use LLM to intelligently reformulate query based on user's answer to clarification"""
    prompt = f"""You are helping to reformulate a user's query based on their answer to a clarification question.

Original Query: "{original_query}"
Clarification Question: "{clarification_question}"
User's Answer: "{user_answer}"

Your task: Reformulate the original query to incorporate the user's answer in a natural, coherent way that preserves the intent.

CRITICAL GUIDELINES:
- PRESERVE ALL UNITS mentioned by user (kgs, tons, lbs, etc.) - DO NOT CHANGE THEM
- If user says "both" or "all", expand to include all mentioned options
- If user specifies a metric, replace generic terms with the specific metric
- If user specifies grouping, add "grouped by" or "by" naturally
- Keep the query concise and clear
- Maintain natural language flow

Examples:

Example 1:
Original: "show weight in kgs"
Clarification: "Which weight metric? Wt/Ft (lbs), WT/Pce (lbs), or Total Wt (Tons)?"
Answer: "Total Wt"
Reformulated: "show Total Wt in kgs"  ← PRESERVED "in kgs"

Example 2:
Original: "list products"
Clarification: "Which attributes? Product ID, Product Description, or both?"
Answer: "both"
Reformulated: "list products showing Product ID and Product Description"

Example 3:
Original: "what's the weight"
Clarification: "Group by Location, Product Type, or show overall total?"
Answer: "by location"
Reformulated: "what's the weight grouped by Location"

Example 4:
Original: "show stock in pounds"
Clarification: "Which stock metric? Qty (PCS) or Current Stock?"
Answer: "current stock"
Reformulated: "show Current Stock in pounds"  ← PRESERVED "in pounds"

Return ONLY the reformulated query as plain text, nothing else.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return response.choices[0].message.content.strip()


def generate_single_agent_clarification(agent_name: str, agent_output: Dict[str, Any], query: str, client: OpenAI) -> str:
    """Generate clarification question for a single specific agent"""
    
    # Extract the actual details from agent detection
    details = agent_output.get('details', 'Ambiguity detected')
    
    prompt = f"""You are helping to clarify an ambiguous spreadsheet query.

User Query: "{query}"

Agent: {agent_output.get('agent_name', agent_name)}
Agent's Analysis: {details}

Your task: Generate ONE clear, concise clarification question based on the agent's analysis.

CRITICAL RULES:
- Use the EXACT column names and values from the agent's analysis
- Do NOT make up example data (Sales, Expenses, Electronics, etc.)
- Extract options from the agent's analysis details
- Keep it conversational and user-friendly
- Ask about ONE thing only

Examples:

Agent Analysis: "Ambiguous column reference: 'weight' could be Wt/Ft (lbs), WT/Pce (lbs), or Total Wt (Tons )"
Question: "Which weight metric would you like? Options: Wt/Ft (lbs), WT/Pce (lbs), or Total Wt (Tons)?"

Agent Analysis: "Query lacks specificity - no location filter specified"
Question: "Which location? Options: PPBC, PPCAL, PPMTL, or all locations?"

Agent Analysis: "Missing grouping criteria"
Question: "How would you like to group the results? Options: by Location, by Product Type, or show overall total?"

Return ONLY the clarification question, nothing else.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()


def detect_followup_query(current_query: str, query_history: list, client: OpenAI) -> dict:
    """Detect if current query is a follow-up to a previous query"""
    if not query_history:
        return {'is_followup': False, 'context': None}
    
    last_query = query_history[-1]
    
    prompt = f"""Analyze if the current query is a follow-up to the previous query.

Previous Query: "{last_query['query']}"
Previous Result Summary: {str(last_query.get('result_summary', 'Data returned'))}

Current Query: "{current_query}"

Determine if the current query is:
1. A follow-up that references the previous result (uses words like "that", "those", "it", "only", "also")
2. A modification/filter of the previous query
3. A completely new, independent query

Respond in JSON format:
{{
  "is_followup": true/false,
  "reference_type": "filter" | "expansion" | "comparison" | "new_query",
  "explanation": "brief explanation",
  "contextualized_query": "rewritten query with context if followup, otherwise same as current"
}}

Examples:

Previous: "total weight by location"
Current: "only PPMTL"
Response: {{"is_followup": true, "reference_type": "filter", "explanation": "User wants to filter previous results to only PPMTL location", "contextualized_query": "show total weight for only PPMTL location"}}

Previous: "show BEAMS"
Current: "what about COILS"
Response: {{"is_followup": false, "reference_type": "new_query", "explanation": "Different product type, new query", "contextualized_query": "what about COILS"}}

Previous: "weight by location"
Current: "show me products at PPBC"
Response: {{"is_followup": false, "reference_type": "new_query", "explanation": "Completely different question", "contextualized_query": "show me products at PPBC"}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    try:
        result = json.loads(response.choices[0].message.content.strip())
        return result
    except:
        return {'is_followup': False, 'context': None, 'contextualized_query': current_query}


def get_filtered_agent_summary(query: str, eclair, cleared_agents: set) -> Dict[str, Any]:
    """Get agent summary but skip already-cleared agents"""
    # Process query but SKIP cleared agents
    result = eclair.process_query_parallel(query, excluded_agents=cleared_agents)
    
    # Extract agent outputs
    agent_outputs = result.get('agent_outputs', {})
    
    # Filter out cleared agents (they're already marked as cleared by process_query_parallel)
    filtered_agents = agent_outputs
    
    return {
        'query': query,
        'agents': filtered_agents,
        'overall_ambiguous': result.get('ambiguous', False),
        'clarification': result.get('clarification', None)
    }


def find_next_ambiguous_agent(summary: Dict[str, Any]) -> Optional[str]:
    """Find the first ambiguous agent that needs clarification"""
    for agent_name, agent_result in summary['agents'].items():
        if agent_result.get('ambiguous', False):
            return agent_name
    return None


# Sidebar
with st.sidebar:
    st.title("🔍 ECLAIR Assistant")
    st.markdown("---")
    
    # Initialize ECLAIR
    if st.button("Initialize System"):
        with st.spinner("Loading ECLAIR system..."):
            st.session_state.eclair = initialize_eclair()
            if st.session_state.eclair:
                st.success("✅ System initialized!")
    
    st.markdown("---")
    
    # Show current query state
    if st.session_state.expanded_query:
        st.subheader("Expanded Query")
        st.info(st.session_state.expanded_query)
        st.metric("Clarifications", st.session_state.clarification_count)
    
    # Show agent status
    if st.session_state.eclair and st.session_state.expanded_query:
        st.subheader("Agent Status")
        all_agents = ['column', 'value', 'aggregation', 'filter']  # Common agent names
        
        for agent in all_agents:
            if agent in st.session_state.cleared_agents:
                st.markdown(f"✅ **{agent.upper()}**: Cleared")
            elif agent == st.session_state.current_ambiguous_agent:
                st.markdown(f"🔴 **{agent.upper()}**: Awaiting answer")
            else:
                st.markdown(f"⏸️ **{agent.upper()}**: Pending")
    
    st.markdown("---")
    
    # Show query history
    if st.session_state.query_history:
        st.subheader("📜 Query History")
        for i, hist in enumerate(reversed(st.session_state.query_history[-3:]), 1):
            with st.expander(f"Query {len(st.session_state.query_history) - i + 1}", expanded=False):
                st.caption(hist['query'])
                st.caption(f"✅ {hist['timestamp'][:19]}")
    
    st.markdown("---")
    
    # Reset button
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.expanded_query = ""
        st.session_state.clarification_count = 0
        st.session_state.cleared_agents = set()
        st.session_state.current_ambiguous_agent = None
        st.session_state.last_clarification = None
        st.session_state.query_history = []
        st.rerun()

# Main chat interface
st.title("💬 ECLAIR Query Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show agent status if available
        if "agent_status" in message:
            with st.expander("🤖 Agent Analysis"):
                for agent, status in message["agent_status"].items():
                    icon = "🔴" if status["ambiguous"] else "🟢"
                    st.markdown(f"**{icon} {agent.upper()}**: {'AMBIGUOUS' if status['ambiguous'] else 'CLEAR'}")

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # Check if ECLAIR is initialized
    if not st.session_state.eclair:
        st.error("Please initialize the system first using the button in the sidebar.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get OpenAI client for all LLM calls
    client = get_openai_client()
    
    # Detect if this is a follow-up query
    followup_info = detect_followup_query(prompt, st.session_state.query_history, client)
    
    # Update expanded query using intelligent reformulation or followup context
    if not st.session_state.expanded_query:
        # First query or fresh query after reset
        if followup_info['is_followup']:
            # Follow-up to previous completed query
            st.session_state.expanded_query = followup_info['contextualized_query']
            with st.sidebar:
                st.info(f"🔗 Detected follow-up: {followup_info['explanation']}")
        else:
            # Brand new query
            st.session_state.expanded_query = prompt
    else:
        # User is answering a clarification - reformulate intelligently
        if st.session_state.last_clarification:
            # Use LLM to merge the answer with original query
            reformulated = reformulate_query_with_answer(
                st.session_state.expanded_query,
                st.session_state.last_clarification,
                prompt,
                client
            )
            st.session_state.expanded_query = reformulated
            st.session_state.clarification_count += 1
            
            # Show reformulation in sidebar for debugging
            with st.sidebar:
                with st.expander("🔄 Query Evolution", expanded=False):
                    st.caption(f"Reformulated to: {reformulated}")
        else:
            # Fallback to concatenation if no clarification stored
            st.session_state.expanded_query += f" {prompt}"
            st.session_state.clarification_count += 1
    
    # Process with ECLAIR
    with st.chat_message("assistant"):
        with st.spinner("Analyzing query..."):
            client = get_openai_client()
            eclair = st.session_state.eclair
            
            # FIRST: If user just answered a question, clear that agent
            if st.session_state.clarification_count > 0 and st.session_state.current_ambiguous_agent:
                # User provided an answer! Clear the agent they just answered
                st.session_state.cleared_agents.add(st.session_state.current_ambiguous_agent)
                st.markdown(f"✅ **{st.session_state.current_ambiguous_agent.upper()}** has been cleared!")
                st.session_state.current_ambiguous_agent = None
            
            # THEN: Get FILTERED agent summary (skip already-cleared agents)
            summary = get_filtered_agent_summary(
                st.session_state.expanded_query, 
                eclair, 
                st.session_state.cleared_agents
            )
            
            # Find next ambiguous agent
            ambiguous_agent = find_next_ambiguous_agent(summary)
            
            if ambiguous_agent:
                # There's an ambiguous agent - generate clarification for THIS agent only
                st.markdown(f"🔍 Checking **{ambiguous_agent.upper()}** agent...")
                
                # Generate clarification for ONLY this specific agent (not all agents)
                agent_output = summary['agents'].get(ambiguous_agent, {})
                
                if agent_output.get('ambiguous', False):
                    # Generate single-agent clarification
                    clarification = generate_single_agent_clarification(
                        ambiguous_agent,
                        agent_output,
                        st.session_state.expanded_query,
                        client
                    )
                    
                    response_text = f"**Clarifying {ambiguous_agent.upper()}:**\n\n{clarification}"
                    st.markdown(response_text)
                    
                    # Store current ambiguous agent and the clarification question
                    st.session_state.current_ambiguous_agent = ambiguous_agent
                    st.session_state.last_clarification = clarification  # Store for reformulation
                    
                    # Store message with agent status
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "agent_status": summary['agents']
                    })
                else:
                    # Agent cleared without needing clarification
                    st.session_state.cleared_agents.add(ambiguous_agent)
                    st.markdown(f"✅ {ambiguous_agent.upper()} is clear!")
                    st.rerun()
            else:
                # ALL CLEAR - generate SQL
                st.markdown("✅ All agents are clear! Generating SQL...")
                
                # Show cleared agents summary
                if st.session_state.cleared_agents:
                    cleared_list = ", ".join([a.upper() for a in st.session_state.cleared_agents])
                    st.caption(f"Cleared agents: {cleared_list}")
                
                # Generate SQL
                sql_code = generate_sql_query(st.session_state.expanded_query, eclair, client)
                
                with st.expander("📝 Generated Code"):
                    st.code(sql_code, language="python")
                
                # Execute query
                st.markdown("🔄 Executing query...")
                results = execute_query(sql_code, eclair.df)
                
                with st.expander("📊 Raw Results"):
                    st.write(results)
                
                # Format results
                st.markdown("✨ Formatting response...")
                final_response = format_results(st.session_state.expanded_query, results, client)
                
                st.markdown("### Answer:")
                st.markdown(final_response)
                
                # Store message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Answer:**\n\n{final_response}\n\n*Based on query: {st.session_state.expanded_query}*"
                })
                
                # Add to query history for cross-query memory
                from datetime import datetime
                st.session_state.query_history.append({
                    'query': st.session_state.expanded_query,
                    'result': results,
                    'result_summary': final_response[:200],  # Store first 200 chars
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
                
                # Keep only last 10 queries to manage memory
                if len(st.session_state.query_history) > 10:
                    st.session_state.query_history = st.session_state.query_history[-10:]
                
                # Reset for next query
                st.session_state.expanded_query = ""
                st.session_state.clarification_count = 0
                st.session_state.cleared_agents = set()
                st.session_state.current_ambiguous_agent = None
                st.session_state.last_clarification = None
