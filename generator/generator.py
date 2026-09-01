from openai import OpenAI


class QwenGenerator:

    def __init__(self):
        self.client = OpenAI(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio"
        )

        self.model = "qwen/qwen2.5-coder-14b"

    def generate(self, prompt):

        response = self.client.chat.completions.create( #Sending the post request to the Local LLM
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content #Returning the first message choice