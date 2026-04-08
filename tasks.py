from environment import CustomerSupportEnv, Action

# ─── Task Definitions ──────────────────────────────────────

TASKS = {
    "easy": {
        "name": "ticket-classify",
        "description": "Classify the support ticket into the correct category.",
        "level": "easy",
        "actions_required": ["classify"],
        "categories": ["billing", "technical", "delivery", "account", "general"],
    },
    "medium": {
        "name": "ticket-classify-priority",
        "description": "Classify the ticket and assign the correct priority level.",
        "level": "medium",
        "actions_required": ["classify", "set_priority"],
        "categories": ["billing", "technical", "delivery", "account", "general"],
        "priorities": ["low", "medium", "high", "critical"],
    },
    "hard": {
        "name": "ticket-full-triage",
        "description": "Classify, prioritize, and draft a reply to the support ticket.",
        "level": "hard",
        "actions_required": ["classify", "set_priority", "draft_reply"],
        "categories": ["billing", "technical", "delivery", "account", "general"],
        "priorities": ["low", "medium", "high", "critical"],
    },
}


# ─── Graders ───────────────────────────────────────────────

def grade_easy(env: CustomerSupportEnv) -> float:
    """
    Easy task grader
    - Correct category = 1.0
    - Wrong category   = 0.0
    """
    if not env.classified:
        return 0.0
    if env.category_given == env.current_ticket["true_category"]:
        return 1.0
    return 0.0


# Priority partial credit map
PRIORITY_PARTIAL = {
    ("critical", "high"): 0.25,
    ("high", "critical"): 0.25,
    ("high", "medium"): 0.25,
    ("medium", "high"): 0.25,
    ("medium", "low"): 0.25,
    ("low", "medium"): 0.25,
}

def grade_medium(env: CustomerSupportEnv) -> float:
    score = 0.0
    if env.classified and env.category_given == env.current_ticket["true_category"]:
        score += 0.5
    if env.prioritized:
        if env.priority_given == env.current_ticket["true_priority"]:
            score += 0.5
        else:
            # Partial credit for close priority
            partial = PRIORITY_PARTIAL.get(
                (env.priority_given, env.current_ticket["true_priority"]), 0.0
            )
            score += partial
    return round(score, 2)


def grade_hard(env: CustomerSupportEnv) -> float:
    score = 0.0
    if env.classified and env.category_given == env.current_ticket["true_category"]:
        score += 0.3
    if env.prioritized:
        if env.priority_given == env.current_ticket["true_priority"]:
            score += 0.3
        else:
            partial = PRIORITY_PARTIAL.get(
                (env.priority_given, env.current_ticket["true_priority"]), 0.0
            )
            score += partial
    if env.replied:
        reply_score = env._grade_reply(env.reply_given)
        score += reply_score * 0.4
    return round(score, 2)


def get_grader(task_level: str):
    """Return the correct grader function for a task level"""
    graders = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }
    return graders.get(task_level, grade_easy)


# ─── Quick Test ────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing all 3 tasks...\n")

    for level in ["easy", "medium", "hard"]:
        print(f"--- Task: {level.upper()} ---")
        env = CustomerSupportEnv(task_level=level)
        obs = env.reset()
        print(f"Ticket: {obs.message}")

        task = TASKS[level]

        # Simulate a correct agent
        if "classify" in task["actions_required"]:
            action = Action(
                action_type="classify",
                value=env.current_ticket["true_category"]
            )
            result = env.step(action)
            print(f"Classify → reward: {result.reward}")

        if "set_priority" in task["actions_required"]:
            action = Action(
                action_type="set_priority",
                value=env.current_ticket["true_priority"]
            )
            result = env.step(action)
            print(f"Priority → reward: {result.reward}")

        if "draft_reply" in task["actions_required"]:
            action = Action(
                action_type="draft_reply",
                value="Sorry for the inconvenience. We will help resolve this issue. Thank you for contacting our team."
            )
            result = env.step(action)
            print(f"Reply    → reward: {result.reward}")

        grader = get_grader(level)
        final_score = grader(env)
        print(f"Final Score: {final_score}\n")