# Document review system
This system is intended to give help to improve educational learning plans. By giving the learning plan as input to the system, user gets knowledge on how to improve the learning plan based on the information (e.g. results from recent articles) from the knowledge base.
## Introduction
This system is based on RAG: when a learning plan is loaded to backend, key technologies are extracted from documents using NER model (see section on NER model below). The extracted technologies are than packed into a single string and further into embedding space using [Snowflake's Arctic-embed-1-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) model. Chunks from knowledge base are extracted based on similarty score with embedded technologies. The system returns 3 pieces of information:
- Technologies extracted from the learning plan itself
- Technologies extracted from relevant (most similiar) knowledge base chunks
- Suggested recommendations for improving learning plan (suggestions are based on parametric GigaChat knowledge)
## Retriever module results
Retriever module is implemented in three steps that further facilitate experiments. The three construction steps are the following:        
- Loading chunks into database without any metadata and then retrieving them with retriever model [Snowflake's Arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) (naive chunking)
- Loading chunks into database with metadata and then retrieving with the same Snowflake's Arctic-embed-l-v2.0
- Loading chunks into database with metadata and then retrieving with Snowflake's Arctic-embed-l-v2.0 and rerank with cross-encoder [ms-macro-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

The following metrics were chosen: 
1. **Precision** (easily interpreted as the ratio of relevant documents in the retrieved documents)
2. **Mean Reciprocal Rank** (shows how close the first relevant document to the first place in retriever results)
3. **Mean average precision** (takes into account both the ratio of relevant documents and their positions relevant to the first position)   
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
Here are the retriever results with summarization metadata for chunking
<table>
  <tr>
    <th></th>
    <th>Mean precision@10</th>
    <th>Mean Average precision@10</th>
    <th>Mean Reciprocal Rank</th>
  </tr>
  <tr>
    <th>Chunk size 300</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Chunk size 600</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Chunk size 1050</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Chunk size 1350</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Chunk size 1800</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>
Here are the retriever results with cross-encoder and summarization
<table>
  <tr>
    <th></th>
    <th>Mean precision@10</th>
    <th>Mean Average precision@10</th>
    <th>Mean Reciprocal Rank</th>
  </tr>
  <tr>
    <th>Chunk size 300</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Chunk size 600</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Chunk size 1050</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Chunk size 1350</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Chunk size 1800</th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

## Reproduction of retriever results
To reproduce naive retrieving results follow these steps:
- Download postgresql from this [official website](https://www.postgresql.org/download/)
- Add postgres bin path to OS environmental variables
- Download naive_chunks_db.sql from this [google drive folder](https://drive.google.com/drive/folders/1wa8_6vqcSkKhPh2rlizgYRxiGkfW18O_?hl=ru)
- Create database and name it "your_database_name"
- Recreate the database using the following script:
```bash
psql -U postgres -d "your_database_name" < naive_chunks_db.sql
```
- Download naive vector dbs from this folder [google_drive_folder](https://drive.google.com/drive/folders/1wa8_6vqcSkKhPh2rlizgYRxiGkfW18O_?hl=ru)
- Make sure you have db password set as DB_PASSWORD environmental variable
- Make sure you have db name set as DB_NAME environmental variable
- Run the Notebook best_naive_chunk_size.ipynb
## NER models results
NER models were trained to extract technologies and concepts from russian university courses syllabi. Syllabi are pdf documents that contain from 20 to 30 pages of information on a specific discipline (each document corresponds to a particular university course). The documents were manually annotated in a BIO format to solve single-entity NER problem. The syllabi dataset can be found [at my HF profile](https://huggingface.co/surpassed). Model evaluations are provided for several lengths of training set (2, 10, 25) to give a perspective on how the number of documents used affects the training time and end results. Models are evaluated using a single validation syllabus document containing 28 pages.


**Evaluation table for training set of length 2**    
The **evaluation** is performened on a single separate syllabus    
The models were trainined on **NVIDIA GeForce GTX 1650** with **4 gb VRAM**
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
