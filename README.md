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
    <th>Chunk size 600</th>
    <td>0.5</td>
    <td>0.432</td>
    <td>0.75</td>
  </tr>
  <tr>
    <th>Chunk size 1050</th>
    <td>0.46</td>
    <td>0.362</td>
    <td>0.657</td>
  </tr>
  <tr>
    <th>Chunk size 1350</th>
    <td>0.495</td>
    <td>0.434</td>
    <td>0.676</td>
  </tr>
  <tr>
    <th>Chunk size 1800</th>
    <td>0.485</td>
    <td>0.409</td>
    <td>0.667</td>
  </tr>
</table>
## Data to reproduce results
Here you can find data to reproduce work results
