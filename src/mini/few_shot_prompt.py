from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain, SequentialChain

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

# # Human ask: Please using Vietnamese to answer this question:What is the antonym of "vui"?
# human_input = "vui"
# final_prompt = few_shot_prompt.format(input=human_input)

# print(final_prompt)

# response = llm.invoke(final_prompt)
# print("\n AI Response")
# print(response.content)

# '''
# Word: vui
# Antonym: buồn, chán nản, ủ rũ, sầu não
# '''
# First chain
antonym_chain = LLMChain(
    llm=llm,
    prompt=few_shot_prompt,
    output_key="raw_antonyms"
)

# Second chain
natural_prompt = PromptTemplate.from_template(
    "Please rewrite the following antonym list into a natural response in Vietnamese:\n"
    "{raw_antonyms}\n"
    "Example: 'The antonyms of vui are buồn, chán nản and beside also have ...'"
)

natural_chain = LLMChain(
    llm=llm,
    prompt=natural_prompt,
    output_key="final_answer"
)

# SequentialChain
overall_chain = SequentialChain(
    chains=[antonym_chain, natural_chain],
    input_variables=["input"], 
    output_variables=["raw_antonyms", "final_answer"]
)

# Testing
result = overall_chain({"input": "vui"})

print("\n Overall Result")
print(result)

print("\n Raw antonyms")
print(result["raw_antonyms"])

print("\n Final natural answer")
print(result["final_answer"])