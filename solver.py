import openai
import time
import collections
from collections import Counter

class GptSolver:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = openai
        self.client.api_key = api_key
        self.model = model
        self.prompt_template = """
            All '*' must be replaced with a single letter or single space.
            Modifying any character other than '*' is PROHIBITED.
            Do NOT correct the grammar.

            Exclude everything except the corrected sentence. You MUST only provide the corrected sentences as the output.

            Given sentence:
            """

    def restore_sentence(self, marked_sentence, num_try=3):
        responses = self._fetch_responses(marked_sentence, num_try)
        final_sentence = self._aggregate_responses(responses)
        return final_sentence

    def _fetch_responses(self, sentence, n):
        responses = []
        while len(responses) < n:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a model that fills in the appropriate single letter for '*'."},
                        {"role": "user", "content": self.prompt_template + sentence},
                    ],
                    temperature=0
                )
                content = response.choices[0].message.content
                responses.append(content)
            except Exception as e:
                print(f"Error: {e}. Retrying in 5s...")
                time.sleep(5)
        return responses

    def _aggregate_responses(self, responses):
        if not responses:
            return ""
        
        lengths = [len(r.split()) for r in responses]
        mode_length = Counter(lengths).most_common(1)[0][0]
        max_len = max(lengths)

        word_count = [collections.Counter() for _ in range(max_len)]
        
        for res in responses:
            words = res.split()
            for j, word in enumerate(words):
                word_count[j][word] += 1
        
        final_words = [wc.most_common(1)[0][0] for wc in word_count[:mode_length]]
        return ' '.join(final_words)