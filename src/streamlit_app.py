import streamlit as st
import requests

st.set_page_config(page_title="MCP OpenMRS Assistant", layout="wide")

st.title("MCP OpenMRS Assistant")

# Input for natural language command/question
query = st.text_area("Enter your natural language command or question:", height=120)

col1, col2 = st.columns([3,1])

with col2:
    if st.button("Generate Preview"):
        if not query.strip():
            st.warning("Please enter a command or question first.")
        else:
            try:
                resp = requests.post("http://localhost:5000/api/query", json={"question": query})
                resp.raise_for_status()
                preview = resp.json()
                st.session_state['latest_preview'] = preview
            except Exception as e:
                st.error(f"Failed to generate preview: {e}")

    if st.button("Confirm"):
        preview = st.session_state.get('latest_preview')
        if not preview:
            st.warning("No preview available to confirm.")
        else:
            payload = { 'command': query }
            if preview.get('sql'):
                payload['sql'] = preview.get('sql')
            if preview.get('action'):
                payload['action'] = preview.get('action')
            if preview.get('_history_id'):
                payload['history_id'] = preview.get('_history_id')
            try:
                resp = requests.post("http://localhost:5000/api/confirm", json=payload)
                resp.raise_for_status()
                result = resp.json()
                st.success("Execution result")
                st.json(result)
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
            st.markdown("**Generated SQL (editable in web UI):**")
            st.code(preview.get('sql', ''), language='sql')
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
                st.markdown("**SQL used for action (preview):**")
                st.code(preview.get('sql', ''), language='sql')
            if preview.get('sql_results'):
                st.markdown("**SQL Preview Results:**")
                st.write(preview.get('sql_results'))

st.sidebar.title("History")
st.sidebar.write("Recent actions and queries are available in the web app sidebar.")