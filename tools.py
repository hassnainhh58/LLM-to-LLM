
# tools.py
import os
import re
import httpx

# CALCULATOR TOOL

def calculator(operation: str, a: float, b: float) -> float:
    """Perform a basic calculation."""
    operation = operation.lower().strip()

    if operation in ["add", "+", "plus"]:
        return a + b
    if operation in ["subtract", "-", "minus"]:
        return a - b
    if operation in ["multiply", "*", "x", "×", "times"]:
        return a * b
    if operation in ["divide", "/", "÷"]:
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    raise ValueError(f"Unknown operation: {operation}")

# WEATHER TOOL

def get_weather(city: str) -> dict:
    """
    Get current weather information for a city.

    Returns weather data on success or an error message
    if the city cannot be found or weather cannot be retrieved.
    """

    # Step 1: Validate the city name
    city = city.strip()

    if (
        not city
        or not any(char.isalpha() for char in city)
    ):
        return {
            "success": False,
            "city": city,
            "error": "Please enter a valid city name.",
        }

    try:
        # Step 2: Find the city's coordinates
        geocoding_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        geocoding_params = {
            "name": city,
            "count": 10,
            "language": "en",
            "format": "json",
        }

        response = httpx.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )

        response.raise_for_status()

        location_data = response.json()
        results = location_data.get("results", [])

        # Step 3: Check whether the city was found
        if not results:
            return {
                "success": False,
                "city": city,
                "error": f"Could not find the city '{city}'.",
            }

        # Step 4: Extract the city name
        city_name = city.split(",")[0].strip()

        # Step 5: Match the returned city name
        location = next(
            (
                result
                for result in results
                if result.get("name", "").casefold()
                == city_name.casefold()
            ),
            None,
        )

        if location is None:
            return {
                "success": False,
                "city": city,
                "error": (
                    f"Could not find a matching city named '{city}'."
                ),
            }

        # Step 6: Get the city's coordinates
        latitude = location["latitude"]
        longitude = location["longitude"]

        # Step 7: Request current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "wind_speed_10m"
            ),
        }

        response = httpx.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )

        response.raise_for_status()

        # Step 8: Extract weather data
        data = response.json()
        current = data.get("current")

        if not current:
            return {
                "success": False,
                "city": city,
                "error": (
                    f"Weather data is unavailable for '{city}'."
                ),
            }

        weather_data = {
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "wind_speed": current["wind_speed_10m"],
        }

        # Step 9: Return successful result
        return {
            "success": True,
            "city": location["name"],
            "weather": weather_data,
        }

    except Exception as error:
        # Step 10: Handle errors
        return {
            "success": False,
            "city": city,
            "error": (
                f"Could not retrieve weather for '{city}'. "
                f"Reason: {error}"
            ),
        }


# WEB SEARCH TOOL

def web_search(question: str) -> str:
    """Search the web using OpenRouter."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('LLM_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": os.getenv("LLM_MODEL"),
        "tools": [{"type": "openrouter:web_search"}],
        "messages": [{"role": "user", "content": question}],
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Guard against error payloads that do not contain choices.
    if not isinstance(data, dict) or not data.get("choices"):
        return f"Error: Unexpected response from the model: {data}"

    return data["choices"][0]["message"]["content"]