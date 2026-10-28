# AI Study Assistant

A small chat program with one **Orchestrator LLM** as the main controller.
The Orchestrator reads your message, splits it into separate questions, and
sends each question to one of two specialist LLMs (Physics or General
Knowledge) or to one of three tools (Calculator, Weather, Web Search).

For example:

> "What is Newton's first law of motion and who is the current PM of Pakistan?"

is split into a **physics** task and a **web_search** task. The answers are
then combined into one reply.

Everything is written with plain Python dictionaries and simple `if` / `elif`
statements - no pydantic and no agent frameworks.

## The files

| File | What it does |
|------|--------------|
| `main.py` | The chat loop. It reads your question, calls the agent, and prints the answer. |
| `agent.py` | The brain. It contains the Orchestrator, the two specialist LLMs, and the tool loop. |
| `tools.py` | The three tools: `calculator`, `get_weather` and `web_search`. |
| `requirements.txt` | The two packages the program needs. |

## Setting it up

1. Install the packages:

   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file next to `main.py`:

   ```
   LLM_API_KEY=your_openrouter_key
   LLM_MODEL=minimax/minimax-m3.1-flash
   ORCHESTRATOR_MODEL=dots-studio/dots-3-note-preview:free
   PHYSICS_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
   GENERAL_MODEL=minimax/minimax-m3.1-flash
   ```

   `ORCHESTRATOR_MODEL`, `PHYSICS_MODEL` and `GENERAL_MODEL` are optional.
   If one is missing, the program uses `LLM_MODEL` instead.

   The example above uses specific free models. Do **not** use
   `openrouter/free` for this project: it is a random router, and it sometimes
   picks a content-safety model that answers `"User Safety: safe"` instead of
   answering your question.

3. Run the program:

   ```
   python main.py
   ```

4. Type `exit` to stop.

## How a message is handled

`run_agent()` in `agent.py` does the whole job in four steps:

1. **Plan.** `plan_tasks()` asks the **Orchestrator LLM** to split the message
   into one or more tasks and choose a route for each task. The Orchestrator
   replies with JSON like this:

   ```json
   {
       "tasks": [
           {"question": "What is Newton's first law of motion?", "route": "physics"},
           {"question": "Who is the current Prime Minister of Pakistan?", "route": "web_search"}
       ]
   }
   ```

   These are the possible routes:

   | Route | Who answers |
   |-------|-------------|
   | `physics` | The Physics LLM |
   | `general` | The General Knowledge LLM |
   | `calculator` | The `calculate` tool |
   | `weather` | The `get_weather_tool` tool |
   | `web_search` | The `web_search_tool` tool |

2. **Answer.** `answer_task()` uses a simple `if` / `elif` to send each task to
   its chosen worker. The Physics and General LLMs answer with text. For a tool
   route, `run_tool_loop()` lets the General LLM call that one tool and then
   explain the result.

3. **Combine.** `combine_results()` puts the sub-answers into one reply. Each
   answer is numbered under its question. The final `topic` and `route` fields
   list every worker that was used, and the `used_*` flags are combined too.

4. **Remember.** Your message and the combined answer are added to
   `message_history`, so the next question can use them as context. Only the
   last `MAX_HISTORY_MESSAGES` messages are sent to the LLMs, which keeps
   prompts small.

`main.py` prints `Handled by: <route>` so you can see the Orchestrator's
choices, for example `Handled by: physics + web_search`.

## The tools

| Tool | Used for |
|------|----------|
| `calculate` | Simple maths with two numbers. |
| `get_weather_tool` | Current weather for one city (Open-Meteo). |
| `web_search_tool` | Facts that change over time (OpenRouter web search). |

The tools themselves in `tools.py` are unchanged from the original version.

## The settings

You can change these at the top of `agent.py`:

| Setting | What it does |
|---------|--------------|
| `MAX_TOOL_ROUNDS` | How many times the model may ask for a tool (default `5`). |
| `MAX_HISTORY_MESSAGES` | How many old messages are sent as context (default `6`). |
| `MAX_TASKS` | The most tasks one message may be split into (default `5`). |
| `MAX_API_ATTEMPTS` | How many times a temporary API error is retried (default `3`). |
| `ORCHESTRATOR_INSTRUCTIONS` | The system prompt that tells the Orchestrator how to split and route. |
| `PHYSICS_INSTRUCTIONS` / `GENERAL_INSTRUCTIONS` | The system prompt given to each specialist. |

## How the answers are written

Every worker is asked for a **short, direct answer** - one or two plain
sentences, with no headings, no lists, and no extra background. The web search
tool is asked for a single sentence that states the fact directly, for example
*"The current Prime Minister of Pakistan is Shehbaz Sharif."*

The rules live in `SHORT_ANSWER_RULES` at the top of `agent.py`, and they are
added to the Physics, General, and Tool prompts.

## The answer format

Every specialist and tool answer must be JSON with these keys:

```json
{
    "answer": "The answer text",
    "topic": "Physics",
    "difficulty": "Beginner",
    "used_calculator": false,
    "used_weather": false,
    "used_web_search": false
}
```

`parse_json_response()` reads that JSON even if the model wraps it in
``` ```json ``` blocks. If the model does not return JSON, the raw text is used
as the answer, so the program still works.

For a single task, the result is returned unchanged. For several tasks,
`combine_results()` creates the final `answer`, `topic`, `difficulty`,
`used_*` flags, and `route`.

## If a model fails

`call_llm()` retries temporary problems such as rate limits (HTTP 429),
temporary server errors (HTTP 502), network timeouts, and empty replies. After
`MAX_API_ATTEMPTS` failed tries, the error is shown by `main.py`.

If a free model is permanently unavailable, change that role's model in
`.env` to another free model that supports tools.
