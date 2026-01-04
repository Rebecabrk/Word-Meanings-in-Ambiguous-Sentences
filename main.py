import requests
import json
import statistics
import random


def call_ollama(prompt: str, temperature: float = 0.7) -> str:
    return requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:8b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
    ).json()["response"]



prompt_template = """
You are a careful linguistic annotator participating in a SemEval-style evaluation.
Your task is to judge how plausible a specific word sense is in a short story, based on human-like narrative understanding.

---

You are given a short story and a specific meaning of an ambiguous word (homonym).
Your task is to evaluate how plausible this meaning is in the given context, as a human reader would.

### Instructions

1. Read the precontext, ambiguous sentence, and ending as one coherent story.
2. Assume the ambiguous sentence uses the homonym with the judged meaning provided.
3. Rate how plausible this interpretation is on a scale from 1 to 5:
   - 1 = completely implausible
   - 2 = mostly implausible
   - 3 = neutral / unclear
   - 4 = plausible
   - 5 = very clearly supported
4. Decide whether the interpretation is nonsensical, meaning it makes the sentence incoherent or absurd in context.

### Output Format (strict)

Return only the following JSON object:

{{
  "choice": <integer from 1 to 5>,
  "nonsensical": <true or false>
}}

### Story Data

Homonym:
{homonym}

Judged Meaning:
{judged_meaning}

Precontext:
{precontext}

Ambiguous Sentence:
{sentence}

Ending:
{ending}
"""


def build_prompt(homonym, judged_meaning, precontext, sentence, ending):
    return prompt_template.format(
        homonym=homonym,
        judged_meaning=judged_meaning,
        precontext=precontext,
        sentence=sentence,
        ending=ending
    )


def parse_response(response: str) -> dict:
    """
    Extract JSON from model output safely.
    Assumes the model returns ONLY JSON, but guards against minor deviations.
    """
    start = response.find("{")
    end = response.rfind("}") + 1
    return json.loads(response[start:end])


def annotate_five_times(prompt: str, runs: int = 5):
    choices = []
    nonsensical = []

    for _ in range(runs):
        temperature = random.uniform(0.6, 0.9)
        raw = call_ollama(prompt, temperature=temperature)
        parsed = parse_response(raw)

        choices.append(parsed["choice"])
        nonsensical.append(parsed["nonsensical"])

    return {
        "choices": choices,
        "average": round(statistics.mean(choices), 2),
        "stdev": round(statistics.stdev(choices), 3),
        "nonsensical": nonsensical
    }


if __name__ == "__main__":
    prompt = build_prompt(
        homonym="track",
        judged_meaning="a pair of parallel rails providing a runway for wheels",
        precontext="The detectives arrived at the abandoned train station. They were looking for signs of the missing artifact. A faint trail caught their attention.",
        sentence="They followed the track.",
        ending="They began to run along the abandoned railway line, hopping from wooden sleeper to sleeper to avoid twisting an ankle."
    )

    result = annotate_five_times(prompt)
    print(json.dumps(result, indent=4))
