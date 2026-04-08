import os
from openai import OpenAI
from environment import CustomerSupportEnv, Action
from tasks import TASKS, get_grader

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

def get_client():
    token = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if token is None:
        raise ValueError("HF_TOKEN or OPENAI_API_KEY environment variable is required")
    return OpenAI(base_url=API_BASE_URL, api_key=token)

SYSTEM_PROMPT = """You are an expert customer support triage agent with 10 years of experience.
You are precise, professional, and always follow the exact output format requested.

You understand these categories deeply:
- billing: payment issues, refunds, double charges, invoices
- technical: app crashes, bugs, loading issues, errors
- delivery: late orders, wrong items, missing orders
- account: login, password, profile, address updates
- general: general questions, how-to, information requests

You understand priority levels:
- low: non-urgent, can wait 24-48 hours (general questions, minor account updates)
- medium: somewhat urgent, handle within a few hours (general complaints)
- high: urgent, handle within 1 hour (payment issues, app not working, wrong orders)
- critical: extremely urgent, handle immediately (order not arrived, customer very distressed)

When drafting replies you always:
- Start with a sincere apology
- Acknowledge the specific issue
- Promise a clear resolution
- End with a polite closing
- Keep it under 3 sentences
"""

FEW_SHOT_EXAMPLES = """
Here are some examples of correct responses:

Example 1:
Ticket: "I was charged twice for order #5678"
Correct: classify(billing)

Example 2:
Ticket: "App crashes every time I open it"
Correct: classify(technical)

Example 3:
Ticket: "My order is 2 hours late and I'm very angry"
Correct: classify(delivery)
Priority: set_priority(critical)

Example 4:
Ticket: "How do I update my email address?"
Correct: classify(account)
Priority: set_priority(low)

Example 5:
Ticket: "Payment failed but money was deducted"
Correct: classify(billing)
Priority: set_priority(high)
Reply: draft_reply(We sincerely apologize for the payment issue. We are investigating the duplicate charge and will resolve it within 2 hours. Thank you for your patience.)
"""

def build_prompt(task_level: str, observation) -> str:
    task = TASKS[task_level]
    base = f"""{FEW_SHOT_EXAMPLES}

Now handle this ticket:
Ticket ID: {observation.ticket_id}
Message: "{observation.message}"

Think step by step:
1. What is the main issue the customer is facing?
2. Which category best fits this issue?
3. How urgent does this seem?

"""
    if task_level == "easy":
        base += f"""Your job is to classify this ticket into exactly one category.
Categories: {", ".join(task["categories"])}

Respond in EXACTLY this format, nothing else:
classify(<category>)

Example: classify(billing)"""

    elif task_level == "medium":
        if not observation.category:
            base += f"""Your job is to classify this ticket into exactly one category.
Categories: {", ".join(task["categories"])}

Respond in EXACTLY this format, nothing else:
classify(<category>)

Example: classify(billing)"""
        else:
            base += f"""Category has been set to: {observation.category}

Now assign a priority level based on urgency.

Priority Guide:
- critical: customer very distressed, immediate action needed
- high: significant problem, needs quick resolution
- medium: moderate issue, handle within few hours
- low: minor issue, can wait

Priorities: {", ".join(task["priorities"])}

Respond in EXACTLY this format, nothing else:
set_priority(<priority>)

Example: set_priority(high)"""

    elif task_level == "hard":
        if not observation.category:
            base += f"""Your job is to classify this ticket into exactly one category.
Categories: {", ".join(task["categories"])}

Respond in EXACTLY this format, nothing else:
classify(<category>)

Example: classify(billing)"""
        elif not observation.priority:
            base += f"""Category has been set to: {observation.category}

Now assign a priority level based on urgency.

Priority Guide:
- critical: customer very distressed, immediate action needed
- high: significant problem, needs quick resolution
- medium: moderate issue, handle within few hours
- low: minor issue, can wait

Priorities: {", ".join(task["priorities"])}

Respond in EXACTLY this format, nothing else:
set_priority(<priority>)

Example: set_priority(high)"""
        else:
            base += f"""Category: {observation.category}
Priority: {observation.priority}

Now draft a short professional reply to this customer.

Your reply MUST include:
- A sincere apology (use words like: sorry, apologize)
- Promise to help resolve the issue (use words like: help, resolve, assist, investigate)
- Be relevant to their specific issue
- End politely (use words like: thank you, regards, team)

Respond in EXACTLY this format, nothing else:
draft_reply(<your reply>)

Example: draft_reply(We sincerely apologize for the inconvenience. Our team will investigate and resolve this issue within the hour. Thank you for your patience.)"""

    return base

def parse_action(response_text: str) -> Action:
    text = response_text.strip()
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("classify("):
            value = line[len("classify("):-1].strip().strip("'\"")
            return Action(action_type="classify", value=value)
        elif line.startswith("set_priority("):
            value = line[len("set_priority("):-1].strip().strip("'\"")
            return Action(action_type="set_priority", value=value)
        elif line.startswith("draft_reply("):
            value = line[len("draft_reply("):-1].strip().strip("'\"")
            return Action(action_type="draft_reply", value=value)
    if "classify(" in text:
        start = text.index("classify(") + len("classify(")
        end = text.index(")", start)
        return Action(action_type="classify", value=text[start:end].strip().strip("'\""))
    elif "set_priority(" in text:
        start = text.index("set_priority(") + len("set_priority(")
        end = text.index(")", start)
        return Action(action_type="set_priority", value=text[start:end].strip().strip("'\""))
    elif "draft_reply(" in text:
        start = text.index("draft_reply(") + len("draft_reply(")
        end = text.rindex(")")
        return Action(action_type="draft_reply", value=text[start:end].strip().strip("'\""))
    return Action(action_type="classify", value="general")

def call_llm(prompt: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1000,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def run_task(task_level: str):
    task = TASKS[task_level]
    env = CustomerSupportEnv(task_level=task_level)
    obs = env.reset()

    print(f"[START] task={task['name']} env=customer-support model={MODEL_NAME}")

    step_num = 0
    last_error = None
    rewards = []

    while not env.done:
        step_num += 1
        try:
            prompt = build_prompt(task_level, obs)
            response_text = call_llm(prompt)
            action = parse_action(response_text)
            result = env.step(action)
            obs = result.observation
            reward = result.reward
            done = result.done
            last_error = result.info.get("error", None)
            rewards.append(reward)
            action_str = f"{action.action_type}('{action.value}')"
            error_str = last_error if last_error else "null"
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_str}")
        except Exception as e:
            last_error = str(e)
            rewards.append(0.0)
            print(f"[STEP] step={step_num} action=null reward=0.00 done=false error={last_error}")
            break

    grader = get_grader(task_level)
    final_score = grader(env)
    success = final_score >= 0.5
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={step_num} score={final_score:.2f} rewards={rewards_str}")
    env.close()
    return final_score

if __name__ == "__main__":
    print("=" * 50)
    print("Customer Support Triage — Baseline Evaluation")
    print("=" * 50)
    scores = []
    for level in ["easy", "medium", "hard"]:
        print(f"\n--- Running {level.upper()} task ---")
        score = run_task(level)
        scores.append(score)
        print(f"Score: {score}")
    avg = sum(scores) / len(scores)
    print(f"\n{'='*50}")
    print(f"Average Score: {avg:.2f}")
    print(f"{'='*50}")