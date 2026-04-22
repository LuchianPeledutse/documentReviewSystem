# Document review system
This system is intended to give help to improve educational learning plans. By giving the learning plan as input to the system, user gets knowledge on how to improve the learning plan based on the information (e.g. results from recent articles) from the knowledge base.
## Introduction
This system is based on RAG: when a learning plan is loaded to backend, key technologies are extracted from documents using NER model (see section on NER model below). The extracted technologies are than packed into a single string and further into embedding space using [Snowflake's Arctic-embed-1-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) model. Chunks from knowledge base are extracted based on similarty score with embedded technologies. The system returns 3 pieces of information:
- Technologies extracted from the learning plan itself
- Technologies extracted from relevant (most similiar) knowledge base chunks
- Suggested recommendations for improving learning plan (suggestions are based on parametric GigaChat knowledge)
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

## Reproduction of retriever results
Here you can find data to reproduce work results

## NER models results
NER models were trained to extract technologies and concepts from russian university courses syllabi. Syllabi are pdf documents that contain from 20 to 30 pages of information on a specific discipline (each document corresponds to a particular university course). The documents were manually annotated in a BIO format to solve single-entity NER problem. The syllabi dataset can be found [at my HF profile](https://huggingface.co/surpassed). Model evaluations are provided for several lengths of training set (2, 10, 25, 50, 100) to give a perspective on how the number of documents used affects the training time and end results. Models are evaluated using a single validation syllabus document containing 28 pages.


**Evaluation table for training set of length 2**
<table>
  <tr>
    <th></th>
    <th>Elman RNN (input_size = 1024, hidden_size = 256, num_layers = 2)</th>
    <th>LSTM (input_size = 1024, hidden_size = 256, num_layers = 1)</th>
    <th>GRU (input_size = 1024, hidden_size = 512, num_layers = 1)</th>
    <th>BERT (input_size = 1024, hidden_size = 256, num_layers = 2)</th>
  </tr>
  <tr>
    <th>Bidirectional = True | F1-score</th>
    <td>0.532</td>
    <td>0.572</td>
    <td>0.574</td>
    <td></td>
  </tr>
  <tr>
    <th>Bidirectional = False | F1-score</th>
    <td>0.505</td>
    <td>0.521</td>
    <td>0.56</td>
    <td>0.506</td>
</tr>
<tr>
    <th>Training Time bi/uni (hrs)</th>
    <td>8.1/0.05</td>
    <td>0.12/0.22</td>
    <td>0.104/3.35</td>
    <td>2.96</td>
</tr>
