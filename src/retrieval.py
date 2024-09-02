from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import pickle
from pyvi.ViTokenizer import tokenize
import numpy as np
from copy import deepcopy
from underthesea import sent_tokenize
import string
from tqdm.notebook import tqdm

# Load dataset
meta_corpus = load_dataset("json", data_files="/data/corpus_chunks.jsonl", split="train").to_list()

def split_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()
    words = [word for word in words if len(word.strip()) > 0]
    return words

# Initialize BM25 retriever
tokenized_corpus = [split_text(doc["passage"]) for doc in tqdm(meta_corpus)]
bm25 = BM25Okapi(tokenized_corpus)

## initiate semantic rertiever
with open('/kaggle/input/wikinlp/corpus_embedding_w150.pkl', 'rb') as f:
    corpus_embs = pickle.load(f)
device = "cuda"
embedder = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder').to(device)

def retrieve(question, topk=50):
    """
    Get most relevant chunks to the question using a combination of BM25 and semantic scores.
    """
    # Initialize query for each retriever (BM25 and semantic)
    tokenized_query = split_text(question)
    segmented_question = tokenize(question)
    question_emb = embedder.encode([segmented_question])
    question_emb /= np.linalg.norm(question_emb, axis=1)[:, np.newaxis]

    # Get BM25 and semantic scores
    bm25_scores = bm25.get_scores(tokenized_query)
    semantic_scores = question_emb @ corpus_embs.T
    semantic_scores = semantic_scores[0]

    # Normalize BM25 scores
    max_bm25_score = max(bm25_scores)
    min_bm25_score = min(bm25_scores)

    def normalize(x):
        return (x - min_bm25_score + 0.1) / (max_bm25_score - min_bm25_score + 0.1)

    # Update chunks' scores
    for i in range(len(meta_corpus)):
        meta_corpus[i]["bm25_score"] = bm25_scores[i]
        meta_corpus[i]["bm25_normed_score"] = normalize(bm25_scores[i])
        meta_corpus[i]["semantic_score"] = semantic_scores[i]

    # Compute combined score (BM25 + semantic)
    for passage in meta_corpus:
        passage["combined_score"] = passage["bm25_normed_score"] * 0.4 + passage["semantic_score"] * 0.6

    # Sort passages by the combined score
    sorted_passages = sorted(meta_corpus, key=lambda x: x["combined_score"], reverse=True)
    return sorted_passages[:topk]

def extract_consecutive_subarray(numbers):
    subarrays = []
    current_subarray = []
    for num in numbers:
        if not current_subarray or num == current_subarray[-1] + 1:
            current_subarray.append(num)
        else:
            subarrays.append(current_subarray)
            current_subarray = [num]

    subarrays.append(current_subarray)  # Append the last subarray
    return subarrays

def merge_contexts(passages):
    passages_sorted_by_id = sorted(passages, key=lambda x: x["id"])

    psg_ids = [x["id"] for x in passages_sorted_by_id]
    consecutive_ids = extract_consecutive_subarray(psg_ids)

    merged_contexts = []
    b = 0
    for ids in consecutive_ids:
        psgs = passages_sorted_by_id[b:b+len(ids)]
        
        # Group passages by title within the consecutive IDs
        title_groups = {}
        for psg in psgs:
            title = psg["title"]
            if title not in title_groups:
                title_groups[title] = []
            title_groups[title].append(psg)
        
        # Merge passages in each title group
        for title, group in title_groups.items():
            if len(group) == 1:
                # If only one passage in the group, add it as is
                psg = group[0]
                merged_contexts.append(dict(
                    title=psg['title'],
                    passage=psg['passage'],
                    score=psg["combined_score"],
                    merged_from_ids=[psg["id"]]
                ))
            else:
                # Merge passages with the same title
                psg_texts = [clean_passage(x) for x in group]
                merged = f"Title: {group[0]['title']}\n\n" + " ".join(psg_texts)
                merged_contexts.append(dict(
                    title=group[0]['title'],
                    passage=merged,
                    score=max([x["combined_score"] for x in group]),
                    merged_from_ids=[x["id"] for x in group]
                ))

        b += len(ids)

    return merged_contexts



def discard_contexts(passages):
    sorted_passages = sorted(passages, key=lambda x: x["score"])
    if len(sorted_passages) == 1:
        return sorted_passages
    else:
        shortened = deepcopy(sorted_passages)
        for i in range(len(sorted_passages) - 1):
            current, next = sorted_passages[i], sorted_passages[i+1]
            if next["score"] - current["score"] >= 0.05:
                shortened = sorted_passages[i+1:]
        return shortened

def expand_context(passage, n_sent=3):
    merged_from_ids = passage["merged_from_ids"]
    title = passage["title"]
    prev_id = merged_from_ids[0] - 1
    next_id = merged_from_ids[-1] + 1

    texts = []
    if prev_id in range(len(meta_corpus)):
        prev_psg = meta_corpus[prev_id]
        if prev_psg["title"] == title:
            prev_text = clean_passage(prev_psg)
            prev_text = " ".join(sent_tokenize(prev_text)[-n_sent:])
            texts.append(prev_text)

    texts.append(clean_passage(passage))

    if next_id in range(len(meta_corpus)):
        next_psg = meta_corpus[next_id]
        if next_psg["title"] == title:
            next_text = clean_passage(next_psg)
            next_text = " ".join(sent_tokenize(next_text)[:n_sent])
            texts.append(next_text)

    expanded_text = " ".join(texts)
    expanded_text = f"Title: {title}\n{expanded_text}"
    new_passage = deepcopy(passage)
    new_passage["passage"] = expanded_text
    return new_passage

def expand_contexts(passages):
    return [expand_context(passage) for passage in passages]

def collapse(passages):
    new_passages = deepcopy(passages)
    titles = {}
    for passage in new_passages:
        title = passage["title"]
        if not titles.get(title):
            titles[title] = [passage]
        else:
            titles[title].append(passage)
    best_passages = []
    for k, v in titles.items():
        best_passage = max(v, key=lambda x: x["score"])
        best_passages.append(best_passage)
    sorted_best_passages = sorted(best_passages, key=lambda x: x["score"], reverse=True)
    return sorted_best_passages

def clean_passage(entry):
    title = entry["title"]
    passage = entry["passage"]
    cleaned_passage = passage[(len(title)+8):].strip()
    return cleaned_passage

def final_clean(passages):
    return "".join(clean_passage(i) for i in passages)

def smooth_contexts(passages):
    """Make the context fed to the LLM better.
    Args:
        passages (list): Chunks retrieved from BM25 + semantic retrieval.

    Returns:
        list: List of whole paragraphs, usually will be more relevant to the initial question.
    """
    # Merge consecutive chunks into one big chunk to ensure continuity
    merged_contexts = merge_contexts(passages)
    # Discard irrelevant contexts
    shortlisted_contexts = discard_contexts(merged_contexts)
    # Expand passages to the whole paragraph that surrounds it
    expanded_contexts = expand_contexts(shortlisted_contexts)
    # Collapse to retain only the paragraph with the highest retrieval score from the same source
    collapsed_contexts = collapse(expanded_contexts)
    final_clean_contexts = final_clean(collapsed_contexts)
    return final_clean_contexts

def get_prompts(question, topk=3):
    PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"  
    top_passages = retrieve(question, topk=topk)
    contexts = smooth_contexts(top_passages)
    instruction = "Dựa vào văn bản sau đây:\n{text}\nHãy trả lời câu hỏi: {question}".format_map({"text": contexts, "question": question})
    input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})   
    return input_prompt