from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.chat_models import ChatOllama

# Examples for few-shot
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "big", "antonym": "small"}
]

# Create a prompt for each example
example_template = "Word: {word}\nAntonym: {antonym}\n"
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template
)

# Prompt few-shot
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Word: {input}\nAntonym:", # suffix for new input
    input_variables=["input"]
)

# Generate the complete prompt for input = "cold"
# print(few_shot_prompt.format(input="cold"))

# Using Mistral
llm = ChatOllama(model="mistral", temperature=0.7)

# Human ask: Please using Vietnamese to answer this question:What is the antonym of "vui"?
human_input = "vui"
final_prompt = few_shot_prompt.format(input=human_input)

print(final_prompt)

response = llm.invoke(final_prompt)
print("\n AI Response")
print(response.content)

'''
Word: vui
Antonym: buồn, chán nản, ủ rũ
'''
