# imports
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import pinecone
from scipy import spatial  # for calculating vector similarities for search
# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 80
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [(row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]



def pinecrone_search(query,model="text-embedding-ada-002",top_k: int = 80):
    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key="b0e7c072-995c-4406-8c41-12238d626882",
        environment="us-west4-gcp"  # find next to API key in console
    )
    index = pinecone.Index('openai')
    # create the query embedding
    xq = openai.Embedding.create(input=query,engine=model)['data'][0]['embedding']
    # query, returning the top 5 most similar results
    res = index.query([xq], top_k=top_k, include_metadata=True)
    strings=[]
    relatednesses=[]
    for match in res['matches']:
        strings.append(match['metadata']['text'])
        relatednesses.append(match['score'])
    return strings,relatednesses


