"""
CLI entry point for Autonomous Research & Insight Generator (API Client).

This CLI communicates with the FastAPI backend for job processing.

Usage:
    python main.py "your research topic"
    python main.py "quantum computing error correction" --papers 5
"""

import sys
import argparse
import time
import requests
import json
from pathlib import Path
from loguru import logger


# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        response.raise_for_status()
        health_data = response.json()
        return True, health_data
    except Exception as e:
        return False, str(e)


def print_progress(stage: str, progress: int, message: str = None):
    """Print progress to console."""
    stages_map = {
        "queued": "⏳ Queued",
        "researching": "📚 Stage 1/3: Researcher",
        "analyzing": "🔍 Stage 2/3: Analyzer",
        "synthesizing": "✍️ Stage 3/3: Synthesizer",
        "complete": "✅ Complete"
    }

    stage_label = stages_map.get(stage, stage)
    bar_length = 40
    filled = int(bar_length * progress / 100)
    bar = "█" * filled + "░" * (bar_length - filled)

    print(f"\r{stage_label} [{bar}] {progress}%", end="", flush=True)

    if message:
        print(f" - {message}", end="", flush=True)


def print_summary(results: dict):
    """Print results summary."""
    print(f"\n\n{'='*70}")
    print(f"  RESEARCH PIPELINE COMPLETE")
    print(f"{'='*70}")

    # Papers summary
    print(f"\n📚 Papers:")
    print(f"  Analyzed: {results['papers_analyzed']}")
    print(f"  Failed: {results['papers_failed']}")

    # Analysis summary
    insights = results['insights_json']
    print(f"\n📊 Analysis:")
    print(f"  Key findings: {len(insights['key_findings'])}")
    print(f"  Contradictions: {len(insights['contradictions'])}")
    print(f"  Complementary: {len(insights['complementary_findings'])}")
    print(f"  Trends: {len(insights['trends'])}")
    print(f"  Research gaps: {len(insights['research_gaps'])}")

    # Outputs
    print(f"\n📄 Outputs:")
    print(f"  Report: outputs/{results['job_id']}/report.md")
    print(f"  Insights: outputs/{results['job_id']}/insights.json")

    print(f"\n{'='*70}\n")


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Autonomous Research & Insight Generator (CLI)"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="LLM agents in enterprise",
        help="Research topic to analyze"
    )
    parser.add_argument(
        "--papers",
        type=int,
        default=5,
        help="Number of papers to fetch (default: 5)"
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  AUTONOMOUS RESEARCH & INSIGHT GENERATOR (CLI)")
    print(f"{'='*70}")
    print(f"\nTopic: {args.topic}")
    print(f"Papers: {args.papers}")
    print(f"Backend: {API_BASE_URL}")
    print(f"\n{'='*70}\n")

    # Check API health
    print("Checking backend API...")
    api_ok, health_info = check_api_health()

    if not api_ok:
        print(f"\n❌ Cannot connect to backend API at {API_BASE_URL}")
        print(f"Error: {health_info}")
        print("\nPlease start the FastAPI backend:")
        print("  uvicorn api.api:app --reload --port 8000")
        sys.exit(1)

    health_data = health_info
    print(f"✅ API connected (Ollama: {'✓' if health_data['ollama_connected'] else '✗'})")
    print(f"Active jobs: {health_data['active_jobs']}\n")

    # Submit research job
    try:
        print(f"Submitting research job...")
        response = requests.post(
            f"{API_BASE_URL}/research",
            json={"topic": args.topic, "max_papers": args.papers},
            timeout=10
        )
        response.raise_for_status()
        job_data = response.json()

        job_id = job_data["job_id"]
        print(f"✅ Job created: {job_id}\n")

    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get("detail", str(e))
        print(f"❌ Failed to create job: {error_detail}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {str(e)}")
        sys.exit(1)

    # Poll job status
    print("Monitoring job progress...\n")
    poll_interval = 3  # seconds
    max_polls = 600  # 30 minutes max
    poll_count = 0

    try:
        while poll_count < max_polls:
            # Get status
            status_response = requests.get(f"{API_BASE_URL}/status/{job_id}", timeout=5)
            status_response.raise_for_status()
            status_data = status_response.json()

            stage = status_data["processing_stage"]
            progress = status_data["progress_percentage"]
            status = status_data["status"]
            message = status_data.get("current_message")

            # Print progress
            print_progress(stage, progress, message)

            # Check if complete
            if status == "complete":
                print("\n")
                break
            elif status == "failed":
                print(f"\n\n❌ Job failed: {status_data.get('error', 'Unknown error')}")
                sys.exit(1)

            # Wait before next poll
            time.sleep(poll_interval)
            poll_count += 1

        if poll_count >= max_polls:
            print("\n\n⏱️ Polling timeout (30 minutes). Job may still be running.")
            print(f"Check status: GET {API_BASE_URL}/status/{job_id}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Job is still running on backend.")
        print(f"Check status: GET {API_BASE_URL}/status/{job_id}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\n\n❌ Error polling status: {str(e)}")
        sys.exit(1)

    # Fetch results
    try:
        print("Fetching results...")
        results_response = requests.get(f"{API_BASE_URL}/results/{job_id}", timeout=10)
        results_response.raise_for_status()
        results = results_response.json()

        # Add job_id to results for summary
        results["job_id"] = job_id

        # Save outputs to job-specific directory
        output_dir = Path("outputs") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        report_path = output_dir / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(results["final_report"])

        # Save insights
        insights_path = output_dir / "insights.json"
        with open(insights_path, "w", encoding="utf-8") as f:
            json.dump(results["insights_json"], f, indent=2, ensure_ascii=False)

        print(f"✅ Results saved\n")

        # Print summary
        print_summary(results)

        print("✅ Research completed successfully!")

    except requests.exceptions.RequestException as e:
        print(f"\n\n❌ Failed to fetch results: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
