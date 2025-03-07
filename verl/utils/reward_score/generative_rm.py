import re
import logging
import ast
import time

import openai

JUDGE_MODEL_NAME = "gpt-4o-2024-08-06"
NULL_COMPLETION = "EMPTY"

HELPFULNESS_PATTERN = re.compile(r"Helpfulness Rating: \[\[(\d+\.?\d*)]]")
RELEVANCE_PATTERN = re.compile(r"Relevance Rating: \[\[(\d+\.?\d*)]]")
DEPTH_PATTERN = re.compile(r"Depth Rating: \[\[(\d+\.?\d*)]]")
CREATIVITY_PATTERN = re.compile(r"Creativity Rating: \[\[(\d+\.?\d*)]]")
LEVEL_OF_DETAILS_PATTERN = re.compile(r"Level of Details Rating: \[\[(\d+\.?\d*)]]")


JUDGE_SYSTEM_PROMPT = "You are a helpful assistant."
JUDGE_META_PROMPT = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
Begin your evaluation by providing a short explanation.
Be as objective as possible.
After providing your explanation, you must rate every attribute of the response on a scale of 1 to 10 by strictly following this format: 
\"Helpfulness Rating: [[helpfulness_rating]]\"
\"Relevance Rating: [[relevance_rating]]\"
\"Depth Rating: [[depth_rating]]\"
\"Creativity Rating: [[creativity_rating]]\"
\"Level of Details Rating: [[level_of_details_rating]]\",

for example:
\"Helpfulness Rating: [[8]]\"
\"Relevance Rating: [[7]]\"
\"Depth Rating: [[4]]\"
\"Creativity Rating: [[3]]\"
\"Level of Details Rating: [[5]]\",

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]
"""

def get_judgement(prompt: str, response: str):
    judge_prompt = JUDGE_META_PROMPT.format(
        question=prompt,
        answer=response,
    )
 
    client = openai.OpenAI()
 
    kwargs = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024,
    }
 
    try:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt},
        ]
        response = client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=messages,
            **kwargs,
        )

        return response.choices[0].message.content
    except (
        openai.RateLimitError,
        openai.InternalServerError,
        openai.UnprocessableEntityError,
    ) as e:
        logging.warning(f"message: {e}")
        time.sleep(1.0)
        return get_judgement(prompt, response)
    except Exception as e:
        logging.warning(f"message: {e}")
        return NULL_COMPLETION


def _parse_judgement(judgement: str, pattern) -> int:
    match = re.search(pattern, judgement)
    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1
 
    return rating


def compute_score(prompt_str: str, response_str: str) -> float:
    """The scoring function for Generative Reward Model.

    Args:
        prompt_str: user instruction or previous chat history
        response_str: the response of the policy
    """

    judgement = get_judgement(prompt_str, response_str)

    helpfulness_score = _parse_judgement(judgement, HELPFULNESS_PATTERN)
    relevance_score = _parse_judgement(judgement, RELEVANCE_PATTERN)
    depth_score = _parse_judgement(judgement, DEPTH_PATTERN)
    creativity_score = _parse_judgement(judgement, CREATIVITY_PATTERN)
    level_of_details_score = _parse_judgement(judgement, LEVEL_OF_DETAILS_PATTERN)

    score = helpfulness_score + relevance_score + depth_score + creativity_score + level_of_details_score

    return float(score)