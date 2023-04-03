# Ranker
## retriever ranker
 - point wise model
   - cross encoder 
     - input : question - {positive passage/context, negative passage/context}
     - ouptut : score {0: positive / 1:negative}

##  data
### train data
|no|제목|출처|비고|
|---|---|---|---|
|1|KorQuAD 1.0|LG CNS||
|2|뉴스 기사 기계독해 데이터|AI HUB||
|3|행정 문서 대상 기계독해 데이터|AI HUB|table mrc는 제외|
|4|도서자료 기계독해|AI HUB||
|5|기계독해|AI HUB||
|5|KLUE MRC|KLUE||
positive context/passage는 annotate 되어 있는 context/passage
negative는 BM25로 가져온 context/passage에서 answer가 없는 context/passage (DPR과 같음)

### inference 시
retrieved context/passage는 100개  
output으로는 5개 

### listwise ranker 
```
[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"context": "...."
	}],
	"positive_ctxs_ids": [..,..,..],
	"retrieved_ctxs": [{
		"title": "...",
		"context": "...."
	},{
		"title": "...",
		"context": "...."
	}, ...],
    "retrieved_ctxs_ids": ["..."]
  }
]
```
n_docs가 만약에 retrieved_ctxs의 개수보다 작다면, sampling

### pointwise ranker(TODO)
```
[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"context": "...."
	}],
	"positive_ctxs_ids": [..,..,..],
	"retrieved_ctxs": [{
		"title": "...",
		"context": "...."
	},{
		"title": "...",
		"context": "...."
	}, ...],
    "retrieved_ctxs_ids": ["..."]
  }
]
```