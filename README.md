# Document review system
This system is intended to give help to improve educational learning plans. By giving the learning plan as input to the system, user gets knowledge on how to improve the learning plan based on the information (e.g. results from recent articles) from the knowledge base.
## Introduction
This system is based no RAG
## Retriever module results
Retriever model [Snowflake's Arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) with naive chunking  
<table>
  <tr>
    <th></th>
    <th>Mean precision@10</th>
    <th>Mean Average precision@10</th>
    <th>Mean Reciprocal Rank</th>
  </tr>
  <tr>
    <th>Chunk size 300</th>
    <td>0.435</td>
    <td>0.326</td>
    <td>0.411</td>
  </tr>
  <tr>
    <th>Row B</th>
    <td>Cell B1</td>
    <td>Cell B2</td>
  </tr>
  <tr>
    <th></th>
  </tr>
</table>
## Data to reproduce results
Here you can find data to reproduce work results
