from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch

# Define the structure and StructuredOutputParser (that will replace parser = StrOutputParser() below)
response_schemas = [
    ResponseSchema(name="word", description="The input word"),
    ResponseSchema(name="antonyms", description="List of antonyms in Vietnamese")
]
structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = structured_parser.get_format_instructions() # Format instructions for the LLM


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

# Prompt few-shot n natural prompt
# few_shot_prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     suffix="Word: {input}\nAntonym:", # suffix for new input
#     input_variables=["input"]
# )
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix=(
        "Word: {input}\nAntonym:\n"
        "{format_instructions}"   # add format instructions
    ),
    input_variables=["input"],
    partial_variables={"format_instructions": format_instructions},
)


natural_prompt = PromptTemplate.from_template(
    "Please rewrite the following antonym list into a natural response in Vietnamese:\n"
    "{raw_antonyms}\n"
    "Example: 'The antonyms of vui are buồn, chán nản and beside also have ...'"
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
antonym_chain_1st = LLMChain(
    llm=llm,
    prompt=few_shot_prompt,
    output_key="raw_antonyms"
)

# Second chain
natural_chain_2nd = LLMChain(
    llm=llm,
    prompt=natural_prompt,
    output_key="final_answer"
)

# # SequentialChain
# overall_chain = SequentialChain(
#     chains=[antonym_chain_1st, natural_chain_2nd],
#     input_variables=["input"], 
#     output_variables=["raw_antonyms", "final_answer"]
# )

# # Testing
# result = overall_chain({"input": "vui"})

# print("\n Overall Result")
# print(result)

# print("\n Raw antonyms")
# print(result["raw_antonyms"])

# print("\n Final natural answer")
# print(result["final_answer"])


# Parser
parser = StrOutputParser()

# Pipeline with pipe operator
antonym_chain = few_shot_prompt | llm | structured_parser # parser
natural_chain = natural_prompt | llm | structured_parser # parser

chain = (
    few_shot_prompt
    | llm
    | parser
    | natural_prompt
    | llm
    | parser
)
# Run
result = chain.invoke({"input": "vui"})
print(result)
'''parser
Word: vui
Antonym: buồn, chán nản, ủ rũ, sầu não
'''

'''structured_parser
{
  "word": "vui",
  "antonyms": ["buồn", "chán nản", "ủ rũ", "sầu não"]
}
'''
# raw_antonyms = antonym_chain.invoke({"input": "vui"})
# final_answer = natural_chain.invoke({"raw_antonyms": raw_antonyms})

# print("\n Raw antonyms")
# print(raw_antonyms)

# print("\n Final natural answer")
# print(final_answer)


##############################################
# Parallel step: run antonym_chain and passthrough input
parallel = RunnableParallel(
    raw_antonyms=antonym_chain,
    original=lambda x: x["input"] # keep original input
)

# Lambda step: build final prompt for natural response
def build_natural_prompt(inputs):
    return natural_prompt.format(raw_antonyms=inputs["raw_antonyms"])

lambda_builder = RunnableLambda(build_natural_prompt)

# Full pipeline
pipeline = parallel | lambda_builder | llm | parser

# Run
result = pipeline.invoke({"input": "vui"})
print("\n FINAL RESULT")
print(result)

'''
Các từ trái nghĩa của "vui" là buồn, chán nản, ủ rũ, sầu não.
'''
##############################################
# Router: if input contains Vietnamese => go to natural_chain
#         else => returns raw_antonyms
def router(inputs):
    if any('\u0100' <= ch <= '\u1EF9' for ch in inputs["original"]):  # check for Vietnamese characters
        return "to_natural"
    return "to_raw"

branch = RunnableBranch(
    (lambda x: router(x) == "to_natural", natural_chain),
    (lambda x: True, lambda x: x["raw_antonyms"])  # default: returns raw_antonyms
)


# Final pipeline
route_pipeline = parallel | branch

result_vi = route_pipeline.invoke({"input": "vui"})
print("\n Vietnamese Input")
print(result_vi)

result_en = route_pipeline.invoke({"input": "happy"})
print("\n English Input")
print(result_en)
