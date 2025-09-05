#!/usr/bin/env python3
"""
coach_orchestrator.py

A lightweight orchestrator for interacting with the CoachKuon LLM via an Ollama server.
This script provides functions to generate marathon training plans and review post‑workout reports.
It includes a demonstration mode that simulates both flows with sample data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# Base URL of the local Ollama API.  Adjust if your server is running elsewhere.
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Name of the model defined in the Modelfile.  If you change the modelfile name, update accordingly.
MODEL_NAME = "coach-kuon-qwen0_5b"


def ask_llm(prompt: str) -> str:
    """
    Sends a prompt to the LLM via the Ollama API and returns the raw response string.

    If the response contains extraneous text (e.g. pre‑amble or trailing content),
    this function attempts to extract the JSON substring.
    """
    data = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=data)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Error contacting Ollama server: {e}") from e
    response_text = resp.json().get("response", "").strip()
    # Attempt to extract JSON if there is any surrounding text.
    if response_text:
        if response_text.lstrip().startswith("{"):
            # It may already be pure JSON.
            return response_text
        # Look for the first and last brace.
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            return response_text[start : end + 1]
    # If unable to find JSON, return the raw text (parsing will fail upstream).
    return response_text


def generate_plan(
    athlete: Dict[str, Any], race: Dict[str, Any], constraints: Optional[Dict[str, Any]] = None, weeks: int = 12
) -> Dict[str, Any]:
    """
    Generates a training plan using the LLM.  The `athlete` and `race` dictionaries
    should include all necessary fields for the prompt.

    :param athlete: Information about the athlete (age, history, weekly_base_km, etc.).
    :param race: Details of the target race (date, course_profile, climate).
    :param constraints: Additional constraints such as number of run days and structure of the plan.
    :param weeks: Number of weeks to generate in the plan.
    :return: Parsed JSON object representing the training plan.
    """
    constraints = constraints or {}
    # Compose the prompt as described in the README.
    prompt_lines = [
        "Reply with VALID JSON for TrainingPlan.json. No extra text.",
        f"Create a {weeks}-week marathon training plan.",
        f"Athlete: {json.dumps(athlete, separators=(',', ':'))}",
        f"Race: {json.dumps(race, separators=(',', ':'))}",
    ]
    if constraints:
        prompt_lines.append(f"Constraints: {json.dumps(constraints, separators=(',', ':'))}")
    prompt_lines.append("Return TrainingPlan.json ONLY.")
    prompt = "\n".join(prompt_lines)

    raw_response = ask_llm(prompt)
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode plan JSON from model response: {raw_response}") from e


def review_session(workout_report: Dict[str, Any], next7_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends a completed workout report and the next seven days of the training plan to the LLM
    and returns the model's recommendations.

    :param workout_report: A dictionary conforming to WorkoutReport.json.
    :param next7_plan: A dictionary representing the next seven days of the plan (a truncated TrainingPlan).
    :return: Parsed JSON object with the model's advice.
    """
    # Compose the prompt for the session reviewer.
    prompt = (
        "Reply with VALID JSON for NextAdvice.json. No extra text.\n"
        f"WORKOUT_REPORT:\n{json.dumps(workout_report, separators=(',', ':'))}\n"
        f"NEXT7_PLAN:\n{json.dumps(next7_plan, separators=(',', ':'))}\n"
        "Rules: be conservative; avoid introducing new gear/fuel within 14 days of race."
    )
    raw_response = ask_llm(prompt)
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode advice JSON from model response: {raw_response}") from e


def load_json_file(path: Path) -> Dict[str, Any]:
    """
    Loads a JSON file from disk.

    :param path: Path to the JSON file.
    :return: Parsed JSON content.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, data: Dict[str, Any]) -> None:
    """
    Writes a dictionary to disk as a JSON file.

    :param path: Path to the JSON file to write.
    :param data: Data to serialize.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_demo() -> None:
    """
    Demonstration that generates a sample plan and simulates a session review.
    """
    # Hypothetical athlete and race information.
    athlete = {
        "age": 32,
        "history": "Completed one marathon two years ago; returning from a minor injury.",
        "weekly_base_km": 40,
        "recent_race": "10km in 50 minutes",
        "injury_notes": "Recovering from knee surgery last year",
    }
    race = {
        "date": "2025-12-01",
        "course_profile": "bridges",
        "climate": "hot-humid",
    }
    constraints = {
        "run_days_per_week": 5,
        "strength_days": 1,
        "long_run_day": "Sun",
        "tempo_day": "Wed",
    }
    print("Generating training plan...")
    try:
        plan = generate_plan(athlete, race, constraints, weeks=12)
    except Exception as e:
        print(f"Error generating plan: {e}", file=sys.stderr)
        return

    print("Plan generated successfully.")
    # Save the plan to disk so users can review it.
    plan_path = Path("demo_training_plan.json")
    write_json_file(plan_path, plan)
    print(f"Training plan saved to {plan_path.resolve()}")

    # Prepare a next7_plan: assume first week is available.
    weeks = plan.get("weeks", [])
    next7_plan = {"weeks": weeks[:1], "race_date": plan.get("race_date"), "notes": plan.get("notes", "")}

    # Simulate a workout report based on the plan.
    workout_report = {
        "athlete": {"age": 32, "sex": "F", "mass_kg": 58},
        "session": {"date": "2025-09-05", "type": "Long", "planned_min": 120},
        "actual": {
            "distance_km": 22.5,
            "duration_min": 125,
            "avg_hr": 155,
            "rpe": 7,
            "elev_gain_m": 150,
            "temp_c": 30,
            "humidity_pct": 85,
        },
        "fueling": {"carbs_g": 65, "sodium_mg": 500, "fluids_ml": 1200},
        "issues": ["mild calf cramps"],
    }

    print("Reviewing session...")
    try:
        advice = review_session(workout_report, next7_plan)
    except Exception as e:
        print(f"Error reviewing session: {e}", file=sys.stderr)
        return
    # Save advice to disk.
    advice_path = Path("demo_next_advice.json")
    write_json_file(advice_path, advice)
    print("Advice received. See output below:\n")
    print(json.dumps(advice, indent=2))
    print(f"\nNext advice saved to {advice_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="CoachKuon orchestrator for training plans and session reviews.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="Run a demonstration of plan generation and session review.")
    group.add_argument("--generate-plan", metavar="ATHLETE_JSON", help="Path to athlete & race JSON input file to generate a plan.")
    group.add_argument("--review-session", nargs=2, metavar=("WORKOUT_JSON", "PLAN_JSON"), help="Paths to workout report JSON and upcoming plan JSON for review.")

    args = parser.parse_args()
    if args.demo:
        run_demo()
        return

    if args.generate_plan:
        input_path = Path(args.generate_plan)
        try:
            input_data = load_json_file(input_path)
        except FileNotFoundError:
            print(f"Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Failed to parse input JSON: {e}", file=sys.stderr)
            sys.exit(1)
        athlete = input_data.get("athlete") or {}
        race = input_data.get("race") or {}
        constraints = input_data.get("constraints") or {}
        weeks = input_data.get("weeks", 12)
        plan = generate_plan(athlete, race, constraints, weeks)
        print(json.dumps(plan, indent=2))
        return

    if args.review_session:
        workout_path = Path(args.review_session[0])
        plan_path = Path(args.review_session[1])
        try:
            workout_report = load_json_file(workout_path)
            next7_plan = load_json_file(plan_path)
        except FileNotFoundError as e:
            print(f"File not found: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}", file=sys.stderr)
            sys.exit(1)
        advice = review_session(workout_report, next7_plan)
        print(json.dumps(advice, indent=2))
        return


if __name__ == "__main__":
    main()