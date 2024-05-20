from langchain_aws.llms import BedrockLLM


def get_text_response(input_content):  # text-to-text client function

    llm = BedrockLLM(  # create a Bedrock llm client
        model_id="cohere.command-text-v14",  # set the foundation model
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0,
            "p": 0.01,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE",
        },
    )
    writing_content = (
        "Give me the cloudformation code for this architecture" + input_content
    )
    return llm.invoke(writing_content)  # return a response to the prompt
