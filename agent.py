
# agent.py

import json
import os
import httpx

from tools import calculator, get_weather, web_search


# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------


ORCHESTRATOR_INSTRUCTIONS = """
You are the Orchestrator of an AI Study Assistant.

Your job is to:
1. Understand the user's complete message.
2. Split it into separate questions when necessary.
3. Choose the best route for each question.

QUESTION UNDERSTANDING RULES:
- Preserve the user's original meaning.
- Do not remove unusual words or phrases just because they
  seem unfamiliar.
- Do not assume that every unfamiliar word is a typo.
- If a question is understandable despite a minor typo, preserve
  its intended meaning.
- If an unclear phrase could change the meaning, keep the
  original wording so the specialist can ask for clarification.
- Do not invent missing information.

Choose exactly one route:
- "physics": Physics questions.
- "general": General knowledge, programming, history, geography,
  and study topics.
- "calculator": Pure arithmetic calculations.
- "weather": Current weather for a city.
- "web_search": Questions requiring current internet information.

If unsure, choose "general".

Reply ONLY with raw JSON:
{"tasks": [{"question": "Complete question", "route": "general"}]}
"""

SHORT_ANSWER_RULES = """
ANSWER STYLE:
- Keep answers short, clear, and professional.
- Answer the question directly.
- Use simple language.
- Avoid unnecessary background information.
- Do not invent facts.
"""

JSON_OUTPUT_INSTRUCTIONS = """
Reply ONLY with a raw JSON object containing these keys:
{
    "answer": "Your clear answer",
    "topic": "Main topic",
    "difficulty": "Beginner, Intermediate, or Advanced",
    "used_calculator": false,
    "used_weather": false,
    "used_web_search": false
}
"""



PHYSICS_INSTRUCTIONS = """
You are a helpful AI Study Assistant specialized in physics.

Before answering, check whether the user's question contains
an unfamiliar word or phrase that could affect its meaning.

Rules:
- Read the entire question carefully.
- Never silently ignore an unfamiliar or unusual phrase.
- Do not assume unfamiliar words are typos.
- If the intended meaning is obvious despite a minor typo,
  answer normally.
- If an unfamiliar phrase could change the meaning, ask
  a brief clarification question.
- If the question is ambiguous, do not present a guessed
  interpretation as certain.
- Answer physics questions correctly and briefly.
- Use simple language.
- Show the key formula when necessary.
- Keep answers short, clear, and professional.

""" + SHORT_ANSWER_RULES + JSON_OUTPUT_INSTRUCTIONS



GENERAL_INSTRUCTIONS = """
You are a helpful AI Study Assistant.
Answer general knowledge, programming, history, geography,
and study questions correctly and briefly.
If you are unsure, say so.
""" + SHORT_ANSWER_RULES + JSON_OUTPUT_INSTRUCTIONS

TOOL_INSTRUCTIONS = """
You are a helpful assistant with access to one tool.
Use the tool to obtain the information or result you need.
Explain the result in one short, clear sentence.
Do not invent tool results.
If a tool returns an error, explain the problem.
""" + SHORT_ANSWER_RULES + JSON_OUTPUT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "add, subtract, multiply, divide, +, -, *, /",
                    },
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["operation", "a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_tool",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": "Search for current or up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Search query",
                    },
                },
                "required": ["question"],
            },
        },
    },
]

TOOLS_BY_ROUTE = {
    "calculator": TOOLS_SCHEMA[0],
    "weather": TOOLS_SCHEMA[1],
    "web_search": TOOLS_SCHEMA[2],
}

VALID_ROUTES = {
    "physics",
    "general",
    "calculator",
    "weather",
    "web_search",
}


# ---------------------------------------------------------------------------
# LLM helper functions
# ---------------------------------------------------------------------------

def get_model(env_name: str) -> str:
    """Get a model name from .env, falling back to LLM_MODEL."""
    return os.getenv(env_name) or os.getenv("LLM_MODEL")


def call_llm(model: str, messages: list, tools: list = None) -> dict:
    """Send a request to OpenRouter and return the assistant message."""

    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise ValueError("LLM_API_KEY is missing from .env")

    if not model:
        raise ValueError("No LLM model is configured in .env")

    payload = {
        "model": model,
        "messages": messages,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    if not isinstance(data, dict) or not data.get("choices"):
        raise RuntimeError(f"Unexpected LLM response: {data}")

    return data["choices"][0]["message"]


def clean_json_text(text: str) -> str:
    """Remove markdown code fences around JSON, if present."""

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def parse_json_object(text: str):
    """Return a dictionary if the text contains a valid JSON object."""

    try:
        data = json.loads(clean_json_text(text))
    except (json.JSONDecodeError, TypeError):
        return None

    return data if isinstance(data, dict) else None


def parse_json_response(text: str, default_answer: str = "") -> dict:
    """Return a consistent answer structure."""

    data = parse_json_object(text)

    if data is None:
        data = {}

    if not data.get("answer"):
        data["answer"] = default_answer or text or "Sorry, I could not generate an answer."

    data.setdefault("topic", "General")
    data.setdefault("difficulty", "Beginner")
    data.setdefault("used_calculator", False)
    data.setdefault("used_weather", False)
    data.setdefault("used_web_search", False)

    return data


# ---------------------------------------------------------------------------
# Task planning
# ---------------------------------------------------------------------------   

def parse_tasks(text: str, user_question: str) -> list:
    """Read the Orchestrator's tasks from its JSON response."""

    data = parse_json_object(text)
    tasks = []

    if data and isinstance(data.get("tasks"), list):
        for item in data["tasks"]:
            if not isinstance(item, dict):
                continue

            question = str(item.get("question", "")).strip()
            route = str(item.get("route", "general")).strip().lower()
            route = route.replace("-", "_").replace(" ", "_")

            if not question:
                continue

            if route not in VALID_ROUTES:
                route = "general"

            tasks.append({
                "question": question,
                "route": route,
            })

    if tasks:
        return tasks

    return [{"question": user_question, "route": "general"}]


def plan_tasks(user_question: str, history: list) -> list:
    """Ask the Orchestrator to split the question and choose routes."""

    messages = (
        [{"role": "system", "content": ORCHESTRATOR_INSTRUCTIONS}]
        + history
        + [{"role": "user", "content": user_question}]
    )

    message = call_llm(get_model("ORCHESTRATOR_MODEL"), messages)

    return parse_tasks(message.get("content", ""), user_question)


# ---------------------------------------------------------------------------
# Specialist models
# ---------------------------------------------------------------------------

def ask_specialist(
    model: str,
    instructions: str,
    user_question: str,
    history: list,
) -> dict:
    """Ask a specialist model to answer a question."""

    messages = (
        [{"role": "system", "content": instructions}]
        + history
        + [{"role": "user", "content": user_question}]
    )

    message = call_llm(model, messages)

    return parse_json_response(message.get("content", ""))


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def run_tool_loop(
    model: str,
    instructions: str,
    user_question: str,
    tools_schema: list,
    history: list,
) -> dict:
    """Let the model call a tool and use its result to answer."""

    messages = (
        [{"role": "system", "content": instructions}]
        + history
        + [{"role": "user", "content": user_question}]
    )

    # Stop after five tool rounds to prevent an endless loop.
    for _ in range(5):
        message = call_llm(model, messages, tools_schema)

        if not message.get("tool_calls"):
            return parse_json_response(message.get("content", ""))

        messages.append(message)

        for tool_call in message["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_call_id = tool_call["id"]

            try:
                arguments = json.loads(
                    tool_call["function"]["arguments"]
                )

                if tool_name == "calculate":
                    result = calculator(
                        arguments["operation"],
                        arguments["a"],
                        arguments["b"],
                    )

                elif tool_name == "get_weather_tool":
                    result = get_weather(arguments["city"])

                elif tool_name == "web_search_tool":
                    result = web_search(arguments["question"])

                else:
                    result = f"Error: Unknown tool {tool_name}"

            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                result = f"Tool error: {error}"

            except Exception as error:
                result = f"Tool error: {error}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result, default=str),
            })

    return parse_json_response(
        "",
        "Sorry, I could not complete the tool operation. Please try again.",
    )


# ---------------------------------------------------------------------------
# Answer individual tasks
# ---------------------------------------------------------------------------

def answer_task(user_question: str, route: str, history: list) -> dict:
    """Send a task to the correct specialist or tool."""

    if route == "physics":
        output_data = ask_specialist(
            get_model("PHYSICS_MODEL"),
            PHYSICS_INSTRUCTIONS,
            user_question,
            history,
        )

    elif route == "general":
        output_data = ask_specialist(
            get_model("GENERAL_MODEL"),
            GENERAL_INSTRUCTIONS,
            user_question,
            history,
        )

    elif route in TOOLS_BY_ROUTE:
        output_data = run_tool_loop(
            get_model("GENERAL_MODEL"),
            TOOL_INSTRUCTIONS,
            user_question,
            [TOOLS_BY_ROUTE[route]],
            history,
        )

    else:
        output_data = ask_specialist(
            get_model("GENERAL_MODEL"),
            GENERAL_INSTRUCTIONS,
            user_question,
            history,
        )
        route = "general"

    output_data["used_calculator"] = route == "calculator"
    output_data["used_weather"] = route == "weather"
    output_data["used_web_search"] = route == "web_search"
    output_data["route"] = route

    return output_data


# ---------------------------------------------------------------------------
# Combine task answers
# ---------------------------------------------------------------------------

def combine_results(task_results: list) -> dict:
    """Combine multiple task answers into one response."""

    if not task_results:
        return {
            "answer": "Sorry, I could not identify a question to answer.",
            "topic": "General",
            "difficulty": "Beginner",
            "used_calculator": False,
            "used_weather": False,
            "used_web_search": False,
            "route": "general",
        }

    if len(task_results) == 1:
        return task_results[0][1]

    answer_parts = []
    topics = []
    difficulties = []
    routes = []

    used_calculator = False
    used_weather = False
    used_web_search = False

    for number, (task, output_data) in enumerate(task_results, start=1):
        answer_parts.append(
            f"{number}. {task['question']}\n"
            f"{output_data.get('answer', '')}"
        )

        topic = output_data.get("topic", "General")

        if topic and topic not in topics:
            topics.append(topic)

        difficulties.append(output_data.get("difficulty", "Beginner"))

        route = output_data.get("route", task["route"])

        if route not in routes:
            routes.append(route)

        used_calculator = (
            used_calculator or bool(output_data.get("used_calculator"))
        )
        used_weather = (
            used_weather or bool(output_data.get("used_weather"))
        )
        used_web_search = (
            used_web_search or bool(output_data.get("used_web_search"))
        )

    if "Advanced" in difficulties:
        difficulty = "Advanced"
    elif "Intermediate" in difficulties:
        difficulty = "Intermediate"
    else:
        difficulty = "Beginner"

    return {
        "answer": "\n\n".join(answer_parts),
        "topic": ", ".join(topics) if topics else "General",
        "difficulty": difficulty,
        "used_calculator": used_calculator,
        "used_weather": used_weather,
        "used_web_search": used_web_search,
        "route": " + ".join(routes),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent(user_question: str, message_history: list) -> tuple:
    """Process a user question and return the answer and updated history."""

    tasks = plan_tasks(user_question, message_history)
    task_results = []

    for task in tasks:
        result = answer_task(
            task["question"],
            task["route"],
            message_history,
        )
        task_results.append((task, result))

    output_data = combine_results(task_results)

    message_history.append({
        "role": "user",
        "content": user_question,
    })

    message_history.append({
        "role": "assistant",
        "content": output_data.get("answer", ""),
    })

    return output_data, message_history
