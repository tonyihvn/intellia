import streamlit as st
import requests

st.set_page_config(page_title="Intelligent Assistant", layout="wide")

st.title("Intelligent Assistant")

# Input for natural language command/question
query = st.text_area("Enter your natural language command or question:", height=120)

col1, col2 = st.columns([3,1])

with col2:
    if st.button("Generate Preview"):
        if not query.strip():
            st.warning("Please enter a command or question first.")
        else:
            try:
                # maintain a simple conversation history in session state so the server
                # can use prior turns as context when classifying/generating
                if 'chat_history' not in st.session_state:
                    st.session_state['chat_history'] = []
                # append the user's turn
                st.session_state['chat_history'].append({"role": "user", "text": query})
                payload = {"question": query, "conversation": st.session_state['chat_history'][-50:]}
                resp = requests.post("http://localhost:5000/api/query", json=payload)
                resp.raise_for_status()
                preview = resp.json()
                st.session_state['latest_preview'] = preview
                # populate editable SQL area if present
                if preview.get('sql'):
                    st.session_state['editable_sql'] = preview.get('sql')
                # append assistant reply into conversation history (best-effort)
                assistant_text = preview.get('response') or preview.get('explanation') or preview.get('sql') or ''
                if assistant_text:
                    st.session_state['chat_history'].append({"role": "assistant", "text": assistant_text})
            except Exception as e:
                st.error(f"Failed to generate preview: {e}")

    if st.button("Confirm"):
        preview = st.session_state.get('latest_preview')
        if not preview:
            st.warning("No preview available to confirm.")
        else:
            payload = { 'command': query }
            # prefer edited SQL in session state
            if st.session_state.get('editable_sql'):
                payload['sql'] = st.session_state.get('editable_sql')
            elif preview.get('sql'):
                payload['sql'] = preview.get('sql')

            if preview.get('action'):
                payload['action'] = preview.get('action')
            if preview.get('_history_id'):
                payload['history_id'] = preview.get('_history_id')
            try:
                # include recent conversation for context during confirm/execution
                if 'chat_history' in st.session_state:
                    payload['conversation'] = st.session_state['chat_history'][-50:]
                resp = requests.post("http://localhost:5000/api/confirm", json=payload)
                resp.raise_for_status()
                result = resp.json()
                st.success("Execution result")
                st.json(result)
                # append execution result to conversation (best-effort)
                exec_text = result.get('message') or result.get('error') or str(result)
                if exec_text and 'chat_history' in st.session_state:
                    st.session_state['chat_history'].append({"role": "assistant", "text": exec_text})
                # refresh history in sidebar
                st.session_state.pop('latest_preview', None)
            except Exception as e:
                st.error(f"Execution failed: {e}")

with col1:
    st.subheader("Preview")
    preview = st.session_state.get('latest_preview')
    if not preview:
        st.info("No preview generated yet. Click 'Generate Preview' after entering your command.")
    else:
        st.write("Type:", preview.get('type'))
        if preview.get('type') == 'query_preview':
            st.markdown("**Generated SQL (editable):**")
            # allow editing of SQL before confirm
            editable_sql = st.text_area("Edit SQL before confirm", value=st.session_state.get('editable_sql', preview.get('sql', '')) , height=200)
            st.session_state['editable_sql'] = editable_sql
            if preview.get('explanation'):
                st.markdown("**Explanation:**")
                st.write(preview.get('explanation'))
            if preview.get('results'):
                st.markdown("**Sample Results:**")
                st.write(preview.get('results'))

        elif preview.get('type') == 'action_preview':
            st.markdown("**Detected Action:**")
            st.write(preview.get('action'))
            if preview.get('sql'):
                st.markdown("**SQL used for action (preview, editable):**")
                editable_sql = st.text_area("Edit SQL before confirm", value=st.session_state.get('editable_sql', preview.get('sql', '')) , height=200)
                st.session_state['editable_sql'] = editable_sql
            if preview.get('sql_results'):
                st.markdown("**SQL Preview Results:**")
                st.write(preview.get('sql_results'))

st.sidebar.title("History")
try:
    resp = requests.get("http://localhost:5000/api/query/history")
    if resp.status_code == 200:
        hist = resp.json()
        if isinstance(hist, list) and hist:
            for h in hist[:20]:
                st.sidebar.write(f"- {h.get('command') or h.get('question')} ({h.get('status') or ''})")
        else:
            st.sidebar.write("No history yet")
except Exception:
    st.sidebar.write("History unavailable")