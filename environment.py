import random
from pydantic import BaseModel
from typing import Optional, List

# Categories that are close enough for partial credit
CATEGORY_PARTIAL = {
    ("general", "account"): 0.5,
    ("account", "general"): 0.5,
    ("general", "billing"): 0.5,
    ("billing", "general"): 0.5,
    ("delivery", "general"): 0.5,
    ("general", "delivery"): 0.5,
}


# ─── Data Models ───────────────────────────────────────────

class Observation(BaseModel):
    ticket_id: str
    message: str
    category: Optional[str] = None
    priority: Optional[str] = None
    reply: Optional[str] = None
    step: int


class Action(BaseModel):
    action_type: str        # "classify", "set_priority", "draft_reply"
    value: str              # the actual value


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


# ─── Ticket Bank ───────────────────────────────────────────

TICKETS = [
    {
        "ticket_id": "T001",
        "message": "I was charged twice for my order #12345. Please refund immediately.",
        "true_category": "billing",
        "true_priority": "high",
        "keywords": ["refund", "charged", "billing"]
    },
    {
        "ticket_id": "T002",
        "message": "How do I reset my password? I can't log in.",
        "true_category": "account",
        "true_priority": "low",
        "keywords": ["password", "login", "account"]
    },
    {
        "ticket_id": "T003",
        "message": "App keeps crashing when I try to make a payment. This has been happening for 2 days.",
        "true_category": "technical",
        "true_priority": "high",
        "keywords": ["crash", "payment", "technical"]
    },
    {
        "ticket_id": "T004",
        "message": "My order hasn't arrived in 3 hours. I have guests at home. Please help urgently!",
        "true_category": "delivery",
        "true_priority": "critical",
        "keywords": ["delivery", "arrived", "urgent"]
    },
    {
        "ticket_id": "T005",
        "message": "Can you tell me what payment methods you accept?",
        "true_category": "general",
        "true_priority": "low",
        "keywords": ["payment", "methods", "general"]
    },
    {
        "ticket_id": "T006",
        "message": "I received the wrong order. I ordered pizza but got burgers.",
        "true_category": "delivery",
        "true_priority": "high",
        "keywords": ["wrong", "order", "delivery"]
    },
    {
        "ticket_id": "T007",
        "message": "Your app is not loading at all. Tried reinstalling but still broken.",
        "true_category": "technical",
        "true_priority": "high",
        "keywords": ["app", "loading", "technical"]
    },
    {
        "ticket_id": "T008",
        "message": "I need to update my delivery address for a future order.",
        "true_category": "account",
        "true_priority": "low",
        "keywords": ["address", "update", "account"]
    },
    {
        "ticket_id": "T009",
        "message": "I have been waiting for my refund for 2 weeks now. This is unacceptable!",
        "true_category": "billing",
        "true_priority": "critical",
        "keywords": ["refund", "waiting", "billing"]
    },
    {
        "ticket_id": "T010",
        "message": "The promo code I applied is not working at checkout.",
        "true_category": "billing",
        "true_priority": "medium",
        "keywords": ["promo", "code", "billing"]
    },
    {
        "ticket_id": "T011",
        "message": "I accidentally placed a duplicate order. Can you cancel one?",
        "true_category": "delivery",
        "true_priority": "high",
        "keywords": ["cancel", "order", "delivery"]
    },
    {
        "ticket_id": "T012",
        "message": "App is showing error code 500 whenever I try to checkout.",
        "true_category": "technical",
        "true_priority": "high",
        "keywords": ["error", "checkout", "technical"]
    },
    {
        "ticket_id": "T013",
        "message": "I want to delete my account and all my data permanently.",
        "true_category": "account",
        "true_priority": "medium",
        "keywords": ["delete", "account", "data"]
    },
    {
        "ticket_id": "T014",
        "message": "My food arrived cold and the packaging was damaged.",
        "true_category": "delivery",
        "true_priority": "high",
        "keywords": ["cold", "damaged", "delivery"]
    },
    {
        "ticket_id": "T015",
        "message": "What are your customer support working hours?",
        "true_category": "general",
        "true_priority": "low",
        "keywords": ["hours", "support", "general"]
    },
    {
        "ticket_id": "T016",
        "message": "I was billed for a subscription I never signed up for!",
        "true_category": "billing",
        "true_priority": "high",
        "keywords": ["billed", "subscription", "billing"]
    },
    {
        "ticket_id": "T017",
        "message": "Push notifications are not working on my iPhone.",
        "true_category": "technical",
        "true_priority": "medium",
        "keywords": ["notifications", "iphone", "technical"]
    },
    {
        "ticket_id": "T018",
        "message": "Delivery person was very rude and unprofessional.",
        "true_category": "delivery",
        "true_priority": "medium",
        "keywords": ["rude", "delivery", "unprofessional"]
    },
    {
        "ticket_id": "T019",
        "message": "How do I add a new credit card to my account?",
        "true_category": "account",
        "true_priority": "low",
        "keywords": ["credit", "card", "account"]
    },
    {
        "ticket_id": "T020",
        "message": "Your website is down. I cannot place any orders!",
        "true_category": "technical",
        "true_priority": "high",
        "keywords": ["website", "down", "technical"]
    },
]


# ─── Environment ───────────────────────────────────────────

class CustomerSupportEnv:
    def __init__(self, task_level: str = "easy"):
        """
        task_level: "easy", "medium", or "hard"
        """
        self.task_level = task_level
        self.current_ticket = None
        self.current_step = 0
        self.done = False
        self.rewards = []

        # Track what agent has done
        self.classified = False
        self.prioritized = False
        self.replied = False

        self.category_given = None
        self.priority_given = None
        self.reply_given = None

    def reset(self) -> Observation:
        """Reset env and return first observation"""
        self.current_ticket = random.choice(TICKETS)
        self.current_step = 0
        self.done = False
        self.rewards = []

        self.classified = False
        self.prioritized = False
        self.replied = False

        self.category_given = None
        self.priority_given = None
        self.reply_given = None

        return Observation(
            ticket_id=self.current_ticket["ticket_id"],
            message=self.current_ticket["message"],
            step=self.current_step
        )

    def step(self, action: Action) -> StepResult:
        """Process one action and return result"""
        self.current_step += 1
        reward = 0.0
        info = {}

        if action.action_type == "classify":
            self.category_given = action.value.lower().strip()
            self.classified = True
            if self.category_given == self.current_ticket["true_category"]:
                reward = 1.0
                info["classify"] = "correct"
            else:
                # Check for partial credit
                partial = CATEGORY_PARTIAL.get(
                    (self.category_given, self.current_ticket["true_category"]), 0.0
                )
                reward = partial
                info["classify"] = f"wrong — expected {self.current_ticket['true_category']}"

        elif action.action_type == "set_priority":
            self.priority_given = action.value.lower().strip()
            self.prioritized = True
            if self.priority_given == self.current_ticket["true_priority"]:
                reward = 1.0
                info["priority"] = "correct"
            else:
                reward = 0.0
                info["priority"] = f"wrong — expected {self.current_ticket['true_priority']}"

        elif action.action_type == "draft_reply":
            self.reply_given = action.value.lower().strip()
            self.replied = True
            reward = self._grade_reply(self.reply_given)
            info["reply_score"] = reward

        else:
            reward = 0.0
            info["error"] = f"Unknown action type: {action.action_type}"

        self.rewards.append(reward)

        # Check if task is done based on level
        self.done = self._check_done()

        obs = Observation(
            ticket_id=self.current_ticket["ticket_id"],
            message=self.current_ticket["message"],
            category=self.category_given,
            priority=self.priority_given,
            reply=self.reply_given,
            step=self.current_step
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=self.done,
            info=info
        )

    def _check_done(self) -> bool:
        """Check if all required actions for task level are done"""
        if self.task_level == "easy":
            return self.classified
        elif self.task_level == "medium":
            return self.classified and self.prioritized
        elif self.task_level == "hard":
            return self.classified and self.prioritized and self.replied
        return False

    def _grade_reply(self, reply: str) -> float:
        """Grade the drafted reply out of 1.0"""
        score = 0.0
        keywords = self.current_ticket["keywords"]

        # Check for apology
        if any(word in reply for word in ["sorry", "apologize", "apologies"]):
            score += 0.25

        # Check for resolution intent
        if any(word in reply for word in ["help", "resolve", "fix", "investigate", "assist"]):
            score += 0.25

        # Check relevance to ticket
        if any(word in reply for word in keywords):
            score += 0.25

        # Check polite closing
        if any(word in reply for word in ["thank", "regards", "sincerely", "team"]):
            score += 0.25

        return round(score, 2)

    def state(self) -> dict:
        """Return current state"""
        return {
            "ticket": self.current_ticket,
            "step": self.current_step,
            "task_level": self.task_level,
            "classified": self.classified,
            "prioritized": self.prioritized,
            "replied": self.replied,
            "rewards": self.rewards,
            "done": self.done
        }

    def close(self):
        """Clean up"""
        pass