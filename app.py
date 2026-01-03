"""
Streamlit UI for Autonomous Research & Insight Generator (API Client).

This frontend communicates with the FastAPI backend for asynchronous job processing.

Usage:
    streamlit run app.py
"""

import streamlit as st
import requests
import time
import json
from datetime import datetime


# API Configuration
API_BASE_URL = "http://localhost:8000"


st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide"
)

# Sidebar: API Status
st.sidebar.title("⚙️ Backend Status")

try:
    health_response = requests.get(f"{API_BASE_URL}/health", timeout=10)
    health_data = health_response.json()

    st.sidebar.markdown(f"**API:** 🟢 Connected")
    st.sidebar.markdown(f"**Ollama:** {'🟢' if health_data['ollama_connected'] else '🔴'} {'Connected' if health_data['ollama_connected'] else 'Disconnected'}")
    st.sidebar.markdown(f"**Active Jobs:** {health_data['active_jobs']}")
    api_available = True
except Exception as e:
    st.sidebar.markdown("**API:** 🔴 Offline")
    st.sidebar.error(f"Cannot connect to backend")
    st.sidebar.error(f"Error: {str(e)}")
    api_available = False

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Backend URL:** `{API_BASE_URL}`")

# Sidebar: Job History
if api_available:
    st.sidebar.markdown("---")
    st.sidebar.title("📚 Job History")

    try:
        jobs_response = requests.get(f"{API_BASE_URL}/jobs", timeout=5)
        jobs_response.raise_for_status()
        jobs_data = jobs_response.json()

        if jobs_data["total_count"] == 0:
            st.sidebar.info("No jobs yet")
        else:
            st.sidebar.markdown(f"*{jobs_data['total_count']} total jobs*")

            # Display each job
            for job in jobs_data["jobs"][:10]:  # Show last 10 jobs
                status_icon = {
                    "complete": "✅",
                    "failed": "❌",
                    "queued": "⏳",
                    "researching": "🔍",
                    "analyzing": "🔬",
                    "comparing": "📊",
                    "synthesizing": "✍️"
                }.get(job["status"], "⚙️")

                # Truncate long topics
                topic_short = job["topic"][:30] + "..." if len(job["topic"]) > 30 else job["topic"]

                with st.sidebar.expander(f"{status_icon} {topic_short}", expanded=False):
                    st.markdown(f"**Job ID:** `{job['job_id'][:8]}...`")
                    st.markdown(f"**Status:** {job['status']}")
                    st.markdown(f"**Created:** {job['created_at'][:19]}")

                    if job["papers_analyzed"] is not None:
                        st.markdown(f"**Papers:** {job['papers_analyzed']} analyzed")

                    if job["error"]:
                        st.error(f"Error: {job['error'][:50]}...")

                    # Load cached results button (only for complete jobs)
                    if job["status"] == "complete":
                        if st.button("📥 Load Results", key=f"load_{job['job_id']}"):
                            st.session_state["selected_job_id"] = job["job_id"]
                            st.rerun()

                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{job['job_id']}"):
                        try:
                            delete_response = requests.delete(f"{API_BASE_URL}/jobs/{job['job_id']}")
                            delete_response.raise_for_status()
                            st.success(f"Deleted job")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {str(e)}")

            if jobs_data["total_count"] > 10:
                st.sidebar.markdown(f"*Showing 10 of {jobs_data['total_count']} jobs*")

    except Exception as e:
        st.sidebar.error(f"Failed to load history: {str(e)}")

# Main UI
st.title("🔬 Autonomous Research Agent")
st.markdown("*Powered by FastAPI + LangGraph + Ollama + FAISS*")

if not api_available:
    st.error("""
    **Backend API is not available!**

    Please start the FastAPI backend:
    ```bash
    cd autonomous-research-agent
    uvicorn api.api:app --reload --port 8000
    ```

    Then refresh this page.
    """)
    st.stop()

# Check if user selected a job from history
if st.session_state.get("selected_job_id"):
    selected_job_id = st.session_state["selected_job_id"]

    st.info(f"📂 Loading cached results for job: `{selected_job_id[:8]}...`")

    try:
        # Fetch cached results
        results_response = requests.get(f"{API_BASE_URL}/results/{selected_job_id}", timeout=10)
        results_response.raise_for_status()
        results = results_response.json()

        # Display Results
        st.markdown("---")
        st.subheader("📊 Research Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Papers Analyzed", results["papers_analyzed"])
        with col2:
            st.metric("Key Findings", len(results["insights_json"]["key_findings"]))
        with col3:
            st.metric("Contradictions", len(results["insights_json"]["contradictions"]))

        # Display Report
        st.markdown("---")
        st.subheader("📄 Research Report")
        st.markdown(results["final_report"])

        # Source Context Inspector
        st.markdown("---")
        with st.expander("🔍 Source Context Inspector (Debugging)", expanded=False):
            st.markdown("**Verify the authenticity of findings** by viewing the raw data used.")

            # Show key findings with their source papers
            if results["insights_json"].get("key_findings"):
                st.subheader("Key Findings Sources")
                for i, finding in enumerate(results["insights_json"]["key_findings"], 1):
                    st.markdown(f"**Finding {i}:**")
                    st.markdown(f"*{finding.get('finding', 'N/A')}*")
                    st.markdown(f"**Source:** {finding.get('citation', 'N/A')}")
                    st.markdown("")

            # Show paper metadata
            st.subheader("Papers Analyzed")
            papers_analyzed = results["papers_analyzed"]
            papers_failed = results["papers_failed"]
            st.markdown(f"- ✅ Successfully processed: **{papers_analyzed}** papers")
            st.markdown(f"- ❌ Failed: **{papers_failed}** papers")

            # Show processing stage info
            st.subheader("Processing Details")
            st.json({
                "topic": results["topic"],
                "job_id": results["job_id"],
                "papers_analyzed": papers_analyzed,
                "created_at": results["created_at"],
                "completed_at": results["completed_at"]
            })

        # Download Buttons
        st.markdown("---")
        st.subheader("💾 Download Results")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=results["final_report"],
                file_name=f"report_{results['topic'].replace(' ', '_')}.md",
                mime="text/markdown"
            )

        with col2:
            st.download_button(
                label="📥 Download Insights (JSON)",
                data=json.dumps(results["insights_json"], indent=2),
                file_name=f"insights_{results['topic'].replace(' ', '_')}.json",
                mime="application/json"
            )

        # Clear selection button
        st.markdown("---")
        if st.button("🔙 Back to New Research"):
            st.session_state.pop("selected_job_id", None)
            st.rerun()

        # Stop here - don't show the input form
        st.stop()

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to load cached results: {str(e)}")
        if st.button("Clear and try again"):
            st.session_state.pop("selected_job_id", None)
            st.rerun()

# Input Section
topic = st.text_input(
    "📝 Research Topic",
    placeholder="e.g., quantum computing error correction"
)

num_papers = st.slider("📚 Number of Papers", 1, 10, 5)

# Start Research Button
if st.button("🚀 Start Research", type="primary", use_container_width=True):
    if not topic:
        st.error("Please enter a research topic")
    else:
        # Submit job to API
        try:
            response = requests.post(
                f"{API_BASE_URL}/research",
                json={"topic": topic, "max_papers": num_papers},
                timeout=10
            )
            response.raise_for_status()
            job_data = response.json()

            job_id = job_data["job_id"]
            st.success(f"✅ Research job created: `{job_id}`")

            # Store job_id in session state for polling
            st.session_state["job_id"] = job_id
            st.session_state["polling_active"] = True
            st.session_state["topic"] = topic

        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json().get("detail", str(e))
            st.error(f"❌ Failed to create job: {error_detail}")
            st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
            st.stop()

# Polling Section (if job is active)
if st.session_state.get("polling_active"):
    job_id = st.session_state["job_id"]
    topic = st.session_state.get("topic", "Unknown")

    # Create placeholders
    status_container = st.container()

    with status_container:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        message_placeholder = st.empty()

    # Poll loop
    max_polls = 600  # 30 minutes max (600 * 3 seconds)
    poll_count = 0

    while st.session_state.get("polling_active") and poll_count < max_polls:
        try:
            # Get job status
            status_response = requests.get(f"{API_BASE_URL}/status/{job_id}", timeout=5)
            status_response.raise_for_status()
            status_data = status_response.json()

            # Update UI based on stage
            stage = status_data["processing_stage"]
            progress = status_data["progress_percentage"]
            status = status_data["status"]

            if stage == "queued":
                status_placeholder.info("⏳ **Queued:** Waiting for job to start...")
            elif stage == "researching":
                status_placeholder.info("📚 **Stage 1/4:** Researcher - Fetching papers from arXiv...")
            elif stage == "analyzing":
                status_placeholder.info("🔍 **Stage 2/4:** Analyzer - Detecting patterns and contradictions...")
            elif stage == "comparing":
                status_placeholder.info("📊 **Stage 3/4:** Comparator - Extracting quantitative metrics...")
            elif stage == "synthesizing":
                status_placeholder.info("✍️ **Stage 4/4:** Synthesizer - Generating final report...")
            elif stage == "complete":
                status_placeholder.success("✅ Research pipeline completed!")
            elif stage == "failed" or status == "failed":
                status_placeholder.error(f"❌ Job failed: {status_data.get('error', 'Unknown error')}")
                st.session_state["polling_active"] = False
                break

            progress_bar.progress(progress / 100)

            if status_data.get("current_message"):
                message_placeholder.markdown(f"*{status_data['current_message']}*")

            # Check if complete
            if status == "complete":
                st.session_state["polling_active"] = False

                # Fetch results
                try:
                    results_response = requests.get(f"{API_BASE_URL}/results/{job_id}", timeout=10)
                    results_response.raise_for_status()
                    results = results_response.json()

                    # Display Results
                    st.markdown("---")
                    st.subheader("📊 Research Summary")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Papers Analyzed", results["papers_analyzed"])
                    with col2:
                        st.metric("Key Findings", len(results["insights_json"]["key_findings"]))
                    with col3:
                        st.metric("Contradictions", len(results["insights_json"]["contradictions"]))

                    # Display Report
                    st.markdown("---")
                    st.subheader("📄 Research Report")
                    st.markdown(results["final_report"])

                    # Source Context Inspector (for debugging/verification)
                    st.markdown("---")
                    with st.expander("🔍 Source Context Inspector (Debugging)", expanded=False):
                        st.markdown("**Verify the authenticity of findings** by viewing the raw data used.")

                        # Show key findings with their source papers
                        if results["insights_json"].get("key_findings"):
                            st.subheader("Key Findings Sources")
                            for i, finding in enumerate(results["insights_json"]["key_findings"], 1):
                                st.markdown(f"**Finding {i}:**")
                                st.markdown(f"*{finding.get('finding', 'N/A')}*")
                                st.markdown(f"**Source:** {finding.get('citation', 'N/A')}")
                                st.markdown("")

                        # Show paper metadata
                        st.subheader("Papers Analyzed")
                        papers_analyzed = results["papers_analyzed"]
                        papers_failed = results["papers_failed"]
                        st.markdown(f"- ✅ Successfully processed: **{papers_analyzed}** papers")
                        st.markdown(f"- ❌ Failed: **{papers_failed}** papers")

                        # Show processing stage info
                        st.subheader("Processing Details")
                        st.json({
                            "topic": results["topic"],
                            "job_id": results["job_id"],
                            "papers_analyzed": papers_analyzed,
                            "created_at": results["created_at"],
                            "completed_at": results["completed_at"]
                        })

                    # Download Buttons
                    st.markdown("---")
                    st.subheader("💾 Download Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.download_button(
                            label="📥 Download Report (Markdown)",
                            data=results["final_report"],
                            file_name=f"report_{topic.replace(' ', '_')}.md",
                            mime="text/markdown"
                        )

                    with col2:
                        st.download_button(
                            label="📥 Download Insights (JSON)",
                            data=json.dumps(results["insights_json"], indent=2),
                            file_name=f"insights_{topic.replace(' ', '_')}.json",
                            mime="application/json"
                        )

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to fetch results: {str(e)}")

                break  # Exit polling loop

            # Wait before next poll (3 seconds)
            time.sleep(3)
            poll_count += 1

            # Force UI update
            st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error polling job status: {str(e)}")
            st.session_state["polling_active"] = False
            break

    if poll_count >= max_polls:
        st.error("⏱️ Polling timeout reached (30 minutes). Job may still be running.")
        st.session_state["polling_active"] = False

# Footer
st.markdown("---")
st.markdown("Built with FastAPI, LangGraph, MCP, and Ollama | No API keys required")
