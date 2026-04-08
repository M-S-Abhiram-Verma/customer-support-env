---
title: Customer Support Triage Environment
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - customer-support
  - reinforcement-learning
---

# 🎧 Customer Support Triage Environment

An OpenEnv-compliant reinforcement learning environment that simulates
real-world customer support triage tasks. An LLM agent must classify
tickets, assign priority levels, and draft professional replies.

## 🌍 Motivation

Customer support triage is a task every tech company performs daily.
Agents must read incoming tickets, understand urgency, route them
correctly, and respond professionally. This environment tests whether
LLM agents can perform this task reliably and accurately.

## 🎯 Tasks

| Task | Level | Description | Max Score |
|------|-------|-------------|-----------|
| ticket-classify | Easy | Classify ticket into correct category | 1.0 |
| ticket-classify-priority | Medium | Classify + assign priority level | 1.0 |
| ticket-full-triage | Hard | Classify + priority + draft reply | 1.0 |

### Difficulty Progression
- **Easy**: Single decision — pick the right category from 5 options
- **Medium**: Two decisions — category + priority level
- **Hard**: Three decisions — category + priority + professional reply

## 🔧 Action Space

| Action | Description | Example |
|--------|-------------|---------|
| `classify(category)` | Classify ticket into a category | `classify(billing)` |
| `set_priority(level)` | Assign urgency level | `set_priority(high)` |
| `draft_reply(text)` | Write a reply to the customer | `draft_reply(We are sorry...)` |

### Categories
`billing` · `technical` · `delivery` · `account` · `general`

### Priority Levels
`low` · `medium` · `high` · `critical`

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | string | Unique ticket identifier |
| `message` | string | Customer's message |
| `category` | string | Category assigned so far |
| `priority` | string | Priority assigned so far |
| `reply` | string | Reply drafted so far |
| `step` | int | Current step number |

## 🏆 Reward Function

| Action | Correct | Partial | Wrong |
|--------|---------|---------|-------|
| classify | 1.0 | 0.5 (close category) | 0.0 |
| set_priority | 1.0 | 0.25 (adjacent level) | 0.0 |
| draft_reply | 0.25–1.0 | based on quality | 0.0 |

Reply scoring breakdown:
- Apology present → +0.25
- Resolution intent → +0.25
- Relevant to issue → +0.25
- Polite closing → +0.25

## 📊 Baseline Scores

Evaluated over 5 episodes using `llama-3.1-8b-instant` via Groq:

| Task | Average | Best | Worst | Std Dev |
|------|---------|------|-------|---------|
| Easy | 1.00 | 1.00 | 1.00 | 0.00 |
| Medium | 0.90 | 1.00 | 0.75 | 0.14 |
| Hard | 1.00 | 1.00 | 1.00 | 0.00 |
| **Overall** | **0.97** | | | |

## 🚀 Setup & Usage

### Local Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/customer-support-env
cd customer-support-env

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_api_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant

# Run inference
python inference.py

# Run full evaluation
python evaluate.py
```

### Docker

```bash
# Build
docker build -t support-env .

# Run
docker run -p 7860:7860 support-env
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/step` | POST | Take an action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks |

#### Example: Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy"}'
```

#### Example: Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "value": "billing"}'
```

## 📁 Project Structure

```
support_env/
├── environment.py    # Core environment logic
├── tasks.py          # Task definitions and graders
├── inference.py      # LLM agent baseline script
├── evaluate.py       # Multi-episode evaluator
├── server.py         # FastAPI server
├── openenv.yaml      # OpenEnv metadata
├── requirements.txt  # Dependencies
├── Dockerfile        # Container config
└── README.md         # Documentation
```

## 🏷️ Tags
`openenv` `customer-support` `triage` `nlp` `llm` `reinforcement-learning`