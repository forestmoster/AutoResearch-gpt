
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
# search function
import search
def num_tokens(text: str, model: str ="gpt-3.5-turbo") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
def query_message(
    query: str,
    token_budget: int,
    model="text-embedding-ada-002",
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings,relatednesses = search.pinecrone_search(query,model=model,)
    introduction = '用下面的文章来回答相关问题，使用中文回答\n'
    question = f"Question:{query}\n"
    message = introduction
    for string in strings:
        next_article = f'"\n{string}\n"'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return question+ message


def ask(
    query: str,
    model: str = "gpt-3.5-turbo",
    token_budget: int = 2000 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "结合下面的文章来回答相关问题，使用中文回答."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

def ask_robot(
    query: str,
    model: str = "gpt-3.5-turbo",
    token_budget: int = 2000 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "结合下面的文章来回答相关问题，使用中文回答."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    response_message = response["choices"][0]["message"]
    return response_message

def query_message_langchain(
    query: str,
    token_budget: int,
    model="text-embedding-ada-002",
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings,relatednesses = search.pinecrone_search(query,model=model,)
    message = ''
    for string in strings:
        next_article = f'"\n{string}\n"'
        if (
            num_tokens(next_article , model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return  message
