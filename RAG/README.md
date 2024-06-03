# RAG (Retrieval-Augmented Generation)

Reference :

```
https://aws.amazon.com/what-is/retrieval-augmented-generation/#:~:text=Retrieval%2DAugmented%20Generation%20(RAG),sources%20before%20generating%20a%20response.
```

```
https://python.langchain.com/v0.1/docs/use_cases/question_answering/
```

### What is Retrieval-Augmented Generation?

RAG extends the already powerful capabilities of LLMs to specific domains or an organization's internal knowledge base, all without the need to retrain the model. It is a cost-effective approach to improving LLM output so it remains relevant, accurate, and useful in various contexts.

### How does Retrieval-Augmented Generation work?

Without RAG, the LLM takes the user input and creates a response based on information it was trained on—or what it already knows. With RAG, an information retrieval component is introduced that utilizes the user input to first pull information from a new data source. The user query and the relevant information are both given to the LLM. The LLM uses the new knowledge and its training data to create better responses

#### Could tell me about RAG ?

A typical RAG application has two main components:
1. Indexing: a pipeline for ingesting data from a source and indexing it. This usually happens offline.
- Load: First we need to load our data. This is done with DocumentLoaders.
- Split: Text splitters break large Documents into smaller chunks. This is useful both for indexing data and for passing it in to a model, since large chunks are harder to search over and won't fit in a model's finite context window.
- Store: We need somewhere to store and index our splits, so that they can later be searched over. This is often done using a VectorStore and Embeddings model.
2. Retrieval and generation: the actual RAG chain, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.
- Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.
- Generate: A ChatModel / LLM produces an answer using a prompt that includes the question and the retrieved data


Reference:
```
https://python.langchain.com/v0.1/docs/use_cases/question_answering/
```
```
https://aws.amazon.com/what-is/retrieval-augmented-generation/
```