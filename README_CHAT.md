# ECLAIR Chat Assistant 🔍

Interactive query clarification system powered by ECLAIR agents.

## Features

- 💬 **Chat Interface**: Natural conversation with your data
- 🤖 **Smart Clarification**: Agents detect ambiguities and ask questions
- 🔄 **Iterative Refinement**: Query gets expanded with each answer
- 📊 **Auto SQL Generation**: Generates queries when everything is clear
- ✨ **Human-Readable Results**: LLM formats results naturally

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key:**
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY='your-api-key-here'
   
   # Linux/Mac
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. **Run the app:**
   ```bash
   streamlit run chat_app.py
   ```

## Usage

1. Click **"Initialize System"** in the sidebar
2. Type your query in the chat (e.g., "What is the total weight?")
3. Answer clarification questions if agents detect ambiguity
4. Get your results when query is clear!

## Example Flow

**User:** "What is the total weight?"

**Agent:** "I need clarification: Which weight metric are you interested in?
- Total Wt (Tons)
- Wt/Ft (lbs)
- WT/Pce (lbs)"

**User:** "Total Wt (Tons)"

**Agent:** "I need clarification: Do you want to group by location, product type, or overall total?"

**User:** "by location"

**Agent:** ✅ Query is clear! 
[Generates SQL, executes, shows results]

## Memory

The app maintains:
- Full conversation history
- Expanded query state
- Clarification count

Use **"Reset Conversation"** to start fresh.
