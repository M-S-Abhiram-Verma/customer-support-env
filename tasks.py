from environment import CustomerSupportEnv, Action

PRIORITY_PARTIAL = {
    ("critical", "high"): 0.25,
    ("high", "critical"): 0.25,
    ("high", "medium"): 0.25,
    ("medium", "high"): 0.25,
    ("medium", "low"): 0.25,
    ("low", "medium"): 0.25,
}

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

def grade_easy(env) -> float:
    try:
        if not hasattr(env, 'classified') or not env.classified:
            return 0.01
        if not hasattr(env, 'current_ticket') or env.current_ticket is None:
            return 0.01
        if not hasattr(env, 'category_given') or env.category_given is None:
            return 0.01
        if env.category_given == env.current_ticket["true_category"]:
            return 0.99
        return 0.01
    except:
        return 0.01

def grade_medium(env) -> float:
    try:
        if not hasattr(env, 'current_ticket') or env.current_ticket is None:
            return 0.01
        score = 0.0
        if hasattr(env, 'classified') and env.classified:
            if hasattr(env, 'category_given') and env.category_given:
                if env.category_given == env.current_ticket["true_category"]:
                    score += 0.5
        if hasattr(env, 'prioritized') and env.prioritized:
            if hasattr(env, 'priority_given') and env.priority_given:
                if env.priority_given == env.current_ticket["true_priority"]:
                    score += 0.49
                else:
                    partial = PRIORITY_PARTIAL.get(
                        (env.priority_given, env.current_ticket["true_priority"]), 0.0
                    )
                    score += partial
        if score <= 0.0:
            return 0.01
        if score >= 1.0:
            return 0.99
        return round(score, 2)
    except:
        return 0.01

def grade_hard(env) -> float:
    try:
        if not hasattr(env, 'current_ticket') or env.current_ticket is None:
            return 0.01
        score = 0.0
        if hasattr(env, 'classified') and env.classified:
            if hasattr(env, 'category_given') and env.category_given:
                if env.category_given == env.current_ticket["true_category"]:
                    score += 0.3
        if hasattr(env, 'prioritized') and env.prioritized:
            if hasattr(env, 'priority_given') and env.priority_given:
                if env.priority_given == env.current_ticket["true_priority"]:
                    score += 0.3
                else:
                    partial = PRIORITY_PARTIAL.get(
                        (env.priority_given, env.current_ticket["true_priority"]), 0.0
                    )
                    score += partial
        if hasattr(env, 'replied') and env.replied:
            if hasattr(env, 'reply_given') and env.reply_given:
                reply_score = env._grade_reply(env.reply_given)
                score += reply_score * 0.38
        if score <= 0.0:
            return 0.01
        if score >= 1.0:
            return 0.99
        return round(score, 2)
    except:
        return 0.01

def get_grader(task_level: str):
    graders = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }
    return graders.get(task_level, grade_easy)

if __name__ == "__main__":
    print("Testing all 3 tasks...\n")
    for level in ["easy", "medium", "hard"]:
        print(f"--- Task: {level.upper()} ---")
        env = CustomerSupportEnv(task_level=level)
        obs = env.reset()
        print(f"Ticket: {obs.message}")
        task = TASKS[level]
        if "classify" in task["actions_required"]:
            result = env.step(Action(action_type="classify", value=env.current_ticket["true_category"]))
            print(f"Classify → reward: {result.reward}")
        if "set_priority" in task["actions_required"]:
            result = env.step(Action(action_type="set_priority", value=env.current_ticket["true_priority"]))
            print(f"Priority → reward: {result.reward}")
        if "draft_reply" in task["actions_required"]:
            result = env.step(Action(action_type="draft_reply", value="Sorry for the inconvenience. We will help resolve this issue. Thank you for contacting our team."))
            print(f"Reply    → reward: {result.reward}")
        grader = get_grader(level)
        print(f"Final Score: {grader(env)}\n")