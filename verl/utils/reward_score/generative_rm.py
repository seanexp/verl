import re
import logging
import ast
import time
from typing import Optional

import openai

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

def get_judgement(prompt: str, response: str, base_url: Optional[str] = None, judge_model_name: str = "gpt-4o-2024-08-06"):
    judge_prompt = JUDGE_META_PROMPT.format(
        question=prompt,
        answer=response,
    )
 
    client = openai.OpenAI(base_url=base_url)
 
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
            model=judge_model_name,
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


def compute_score(
    prompt_str: str,
    response_str: str,
    base_url: Optional[str] = None,
    judge_model_name: str = "gpt-4o-2024-08-06"
) -> float:
    """The scoring function for Generative Reward Model.

    Args:
        prompt_str: user instruction or previous chat history
        response_str: the response of the policy
    """

    judgement = get_judgement(prompt_str, response_str, base_url, judge_model_name)

    helpfulness_score = _parse_judgement(judgement, HELPFULNESS_PATTERN)
    relevance_score = _parse_judgement(judgement, RELEVANCE_PATTERN)
    depth_score = _parse_judgement(judgement, DEPTH_PATTERN)
    creativity_score = _parse_judgement(judgement, CREATIVITY_PATTERN)
    level_of_details_score = _parse_judgement(judgement, LEVEL_OF_DETAILS_PATTERN)

    score = helpfulness_score + relevance_score + depth_score + creativity_score + level_of_details_score

    return float(score)


if __name__ == "__main__":
    prompt_str = '<|user|>\njava 로 URL의 이미지를  다운로드 받는 어플리케이션을 만들려고 해<|eot_id|>\n<|assistant|>\nJava를 사용하여 URL에서 이미지를 다운로드하는 어플리케이션을 만드는 방법에 대해 안내해드리겠습니다.\n\n1. URL 객체 생성하기\nJava에서 URL을 다루기 위해서는 java.net 패키지의 URL 클래스를 사용합니다. URL 클래스를 사용하여 이미지를 다운로드 할 URL을 지정합니다. 예를 들어, "http://example.com/image.jpg"라는 이미지 URL을 다운로드하려면 다음과 같은 코드를 사용합니다.\n\n```java\nURL imageUrl = new URL("http://example.com/image.jpg");\n```\n\n2. HttpURLConnection 객체 생성하기\nURL 객체를 사용하여 이미지를 다운로드 하기 위해서는 HttpURLConnection 클래스를 사용합니다. URL.openConnection() 메소드를 사용하여 HttpURLConnection 객체를 생성합니다.\n\n```java\nHttpURLConnection connection = (HttpURLConnection) imageUrl.openConnection();\n```\n\n3. 이미지 다운로드\n이미지를 다운로드하려면 HttpURLConnection 객체에서 InputStream을 가져와서 이미지 데이터를 읽습니다. 다음은 이미지 데이터를 저장할 OutputStream 객체를 생성하고 이미지를 다운로드하는 예제 코드입니다.\n\n```java\nInputStream inputStream = connection.getInputStream();\nOutputStream outputStream = new FileOutputStream("image.jpg");\n\nbyte[] buffer = new byte[2048];\nint length;\nwhile ((length = inputStream.read(buffer)) != -1) {\n    outputStream.write(buffer, 0, length);\n}\n\noutputStream.close();\ninputStream.close();\n```\n\n위의 코드에서 "image.jpg"는 다운로드 한 이미지를 저장할 파일 이름입니다.\n\n4. 완성된 코드\n\n위의 단계들을 합치면 다음과 같이 이미지를 다운로드하는 Java 코드를 작성할 수 있습니다.\n\n```java\nimport java.io.*;\nimport java.net.*;\n\npublic class ImageDownloader {\n    public static void main(String[] args) throws Exception {\n        URL imageUrl = new URL("http://example.com/image.jpg");\n        HttpURLConnection connection = (HttpURLConnection) imageUrl.openConnection();\n\n        InputStream inputStream = connection.getInputStream();\n        OutputStream outputStream = new FileOutputStream("image.jpg");\n\n        byte[] buffer = new byte[2048];\n        int length;\n        while ((length = inputStream.read(buffer)) != -1) {\n            outputStream.write(buffer, 0, length);\n        }\n\n        outputStream.close();\n        inputStream.close();\n    }\n}\n```\n\n위의 코드를 실행하면 "image.jpg" 파일이 현재 작업 디렉토리에 생성되고, 해당 파일에 이미지 데이터가 저장됩니다.\n\n참고: 위의 코드에서는 이미지 데이터를 한 번에 읽어서 저장하지만, 만약 이미지가 매우 큰 경우 OutOfMemoryError가 발생할 수 있으므로, 이미지를 작은 조각들로 나누어서 저장하는 방법도 있습니다.<|eot_id|>\n<|user|>\nhttp, https 를 지원하고 한글도메인 한글파일명도 지원하고 리다이렉트 응답도 지원해야해<|eot_id|>\n'
    response_str = 'Java를 사용하여 HTTP, HTTPS를 지원하고 한글 도메인 및 한글 파일명, 리다이렉트를 지원하도록 이미지를 다운로드하는 어플리케이션을 작성하려면 다음과 같이 할 수 있습니다. \n\n이를 위해 필수적으로 아래와 같은 기능들을 구현해야 합니다:\n\n1. URL 인코딩/디코딩 (한글 도메인 및 파일명 지원)\n2. HttpURLConnection 설정 (HTTPS 지원)\n3. 리다이렉트 처리를 위한 코드\n4. 예외 처리 및 리소스 최종화 (try-with-resources 사용)\n\n코드 예제를 제공합니다:\n\n```java\nimport java.io.*;\nimport java.net.*;\n\npublic class ImageDownloader {\n    public static void main(String[] args) {\n        try {\n            String urlString = "http://example.com/한글파일.jpg";\n            downloadImage(urlString, "다운로드된파일.jpg");\n        } catch (Exception e) {\n            e.printStackTrace();\n        }\n    }\n\n    public static void downloadImage(String urlString, String outputFileName) throws IOException {\n        // URL 인코딩 (한글 도메인 및 파일명을 지원하기 위해)\n        URL url = new URL(encodeUrl(urlString));\n\n        HttpURLConnection connection = (HttpURLConnection) url.openConnection();\n        connection.setInstanceFollowRedirects(false); // 리다이렉트 수동 처리\n        \n        int status = connection.getResponseCode();\n        if (status == HttpURLConnection.HTTP_MOVED_TEMP || \n            status == HttpURLConnection.HTTP_MOVED_PERM || \n            status == HttpURLConnection.HTTP_SEE_OTHER) {\n\n            // 리다이렉트 처리\n            String newUrl = connection.getHeaderField("Location");\n            downloadImage(newUrl, outputFileName);\n            return;\n        }\n\n        // 이미지 파일 다운로드\n        try (InputStream inputStream = connection.getInputStream();\n             OutputStream outputStream = new FileOutputStream(outputFileName)) {\n            byte[] buffer = new byte[2048];\n            int length;\n            while ((length = inputStream.read(buffer)) != -1) {\n                outputStream.write(buffer, 0, length);\n            }\n        }\n    }\n\n    // URL 인코딩 (한글 도메인 및 파일명을 지원하기 위해)\n    private static String encodeUrl(String urlString) throws MalformedURLException, UnsupportedEncodingException {\n        URL url = new URL(urlString);\n        String protocol = url.getProtocol();\n        String host = IDN.toASCII(url.getHost());\n        int port = url.getPort();\n        String path = url.getPath();\n        String query = url.getQuery();\n\n        StringBuilder encodedUrl = new StringBuilder();\n        encodedUrl.append(protocol).append("://").append(host);\n        if (port != -1) {\n            encodedUrl.append(\':\').append(port);\n        }\n        encodedUrl.append(URLEncoder.encode(path, "UTF-8")\n                        .replaceAll("%2F", "/")) // 슬래시를 다시 원본으로\n                  .append(query == null ? "" : "?" + URLEncoder.encode(query, "UTF-8"));\n\n        return encodedUrl.toString();\n    }\n}\n```\n\n### 주요 포인트:\n\n1. **URL 인코딩**:\n   - 한글 도메인과 파일명을 지원하기 위해 `IDN.toASCII()`와 `URLEncoder.encode()`를 사용하여 도메인과 경로를 인코딩합니다.\n\n2. **리다이렉트 처리**:\n   - `HttpURLConnection`을 설정하여 수동으로 리다이렉트를 처리하도록 합니다.\n   - `3XX` 리다이렉트 응답 코드를 확인한 후 새로운 위치에서 다운로드를 시도합니다.\n\n3. **HTTPS 지원**:\n   - `HttpURLConnection`을 사용하면 HTTPS도 자동으로 지원됩니다.\n\n4. **자원 관리**:\n    - `try-with-resources` 구문을 사용하여 InputStream과 OutputStream을 자동으로 닫아줍니다.\n\n위 코드를 실행하면, 한글 도메인 및 파일명을 지원하는 이미지를 다운로드하고 리다이렉트를 처리하여 올바른 파일을 저장하는 데 문제가 없어야 합니다.'
    base_url = "http://instance-24179.prj-875:8000/v1"
    judge_model_name = "Qwen2.5-72B-Instruct"
    score = compute_score(
        prompt_str=prompt_str,
        response_str=response_str,
        base_url=base_url,
        judge_model_name=judge_model_name,
    )

    print(score)