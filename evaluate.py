import os
import statistics
from environment import CustomerSupportEnv, Action
from tasks import get_grader
from inference import run_task, build_prompt, parse_action, call_llm, MODEL_NAME

# ─── Multi Episode Evaluator ───────────────────────────────

def run_multiple_episodes(task_level: str, num_episodes: int = 5):
    """Run multiple episodes and report average score"""
    print(f"\n{'='*50}")
    print(f"Running {num_episodes} episodes for {task_level.upper()} task")
    print(f"{'='*50}")

    scores = []
    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---")
        score = run_task(task_level)
        scores.append(score)
        print(f"Episode {episode} Score: {score}")

    # Stats
    avg = round(max(0.01, min(0.99, statistics.mean(scores))), 2)
    best = max(scores)
    worst = min(scores)

    if len(scores) > 1:
        std = statistics.stdev(scores)
    else:
        std = 0.01

    print(f"\n📊 Results for {task_level.upper()} over {num_episodes} episodes:")
    print(f"   Average : {avg:.2f}")
    print(f"   Best    : {best:.2f}")
    print(f"   Worst   : {worst:.2f}")
    print(f"   Std Dev : {std:.2f}")

    return {
        "task_level": task_level,
        "episodes": num_episodes,
        "scores": scores,
        "average": round(avg, 2),
        "best": round(best, 2),
        "worst": round(worst, 2),
        "std_dev": round(std, 2)
    }


def run_full_evaluation(num_episodes: int = 5):
    """Run full evaluation across all task levels"""
    print(f"\n{'='*50}")
    print(f"FULL EVALUATION — {num_episodes} episodes per task")
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*50}")

    all_results = []
    all_scores = []

    for level in ["easy", "medium", "hard"]:
        result = run_multiple_episodes(level, num_episodes)
        all_results.append(result)
        all_scores.extend(result["scores"])

    # Overall stats
    overall_avg = round(max(0.01, min(0.99, statistics.mean(all_scores))), 2)

    print(f"\n{'='*50}")
    print(f"📊 OVERALL SUMMARY")
    print(f"{'='*50}")
    for result in all_results:
        print(f"{result['task_level'].upper():8} → avg: {result['average']:.2f} | best: {result['best']:.2f} | worst: {result['worst']:.2f} | std: {result['std_dev']:.2f}")
    print(f"{'─'*50}")
    print(f"{'OVERALL':8} → avg: {overall_avg:.2f}")
    print(f"{'='*50}")

    return all_results


if __name__ == "__main__":
    # Run 5 episodes per task level
    run_full_evaluation(num_episodes=5)