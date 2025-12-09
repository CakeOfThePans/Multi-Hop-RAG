SINGLE_HOP_SYSTEM_PROMPT = (
    "You are a precise QA assistant. Return just the short answer phrase with no explanation, and no full sentences."
    "If you are COMPLETELY UNSURE of the answer based on the provided passages, respond with 'Unknown'."
)

def build_singlehop_user_prompt(question, passages):
    bundle = "\n\n".join([f"PASSAGE {i+1}:\n{p}" for i, p in enumerate(passages)])
    return f"{bundle}\n\nQUESTION: {question}\nANSWER:"

DECOMPOSER_SYSTEM_PROMPT = (
    "You have to break complex questions into concise, sequential sub-questions."
)

DECOMPOSER_USER_TEMPLATE = (
    "- Produce BETWEEN 1 and {max_subqs} sub-questions that will help answer the main question.\n"
    "- Each sub-question MUST be under 18 words.\n"
    "- Each sub-question must be specific and answerable using a retrieval system.\n"
    "- Each sub-question must contribute useful information towards answering the main question.\n"
    "- Answers to sub-questions must solve the main question when combined.\n"
    "- Order the list so answering in order solves the original question.\n"
    "- No extra keys. No commentary. No markdown.\n\n"
    "QUESTION: {q}"
)

COMPOSE_QUERY_SYSTEM_PROMPT = (
    "You are to rewrite questions into focused search queries for retrieval."
)

ANSWER_SUBQ_SYSTEM_PROMPT = (
    "You are a precise QA assistant. Return only the short answer phrase. "
    "No explanation, no full sentences."
)

FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are a precise QA assistant. Return only the short answer phrase "
    "(in some cases 1–2 words will suffice). No explanation, no full sentences."
)


def build_compose_query_prompt(original_question, subq, hops):
    
    mem_lines = "\n".join([f"{i+1}. {h['subq']} -> {h['answer']}" for i, h in enumerate(hops)]) or "None yet."
    
    return (
        "Rewrite the following sub-question into a concise search query for document retrieval.\n"
        "- use entities/names filled by PRIOR ANSWERS when relevant.\n"
        "- if prior answers don't help, keep the original sub-question details.\n"
        "- Keep it under 18 words. No pronouns like this/that/it.\n"
        "- The query must remain as a question.\n"
        "- Be specific and retrieval-friendly (names, years, titles).\n\n"
        f"ORIGINAL QUESTION: {original_question}\n\n"
        f"PRIOR ANSWERS:\n{mem_lines}\n\n"
        f"SUB-QUESTION: {subq}\n\n"
        "SEARCH QUERY:"
    )

def build_answer_subq_prompt(original_question, subq, passages, hops):
    
    mem_lines = "\n".join([f"{i+1}. {h['subq']} -> {h['answer']}" for i, h in enumerate(hops)]) or "None."
    ctx = "\n\n".join([f"PASSAGE {i+1}:\n{p}" for i, p in enumerate(passages)])
    
    return (
        "Answer the CURRENT SUB-QUESTION using the CONTEXT passages provided.\n"
        "If you are unsure or the context does not provide enough information, respond with 'Unknown'.\n"
        f"ORIGINAL QUESTION:\n{original_question}\n\n"
        f"PRIOR ANSWERS:\n{mem_lines}\n\n"
        f"CURRENT SUB-QUESTION:\n{subq}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        "Answer (short phrase only):"
    )

def build_final_answer_prompt(original_question, hops):
    
    mem_lines = "\n".join([f"{i+1}. {h['subq']} -> {h['answer']}" for i, h in enumerate(hops)]) or "None."
    support = []


    for i, h in enumerate(hops):
        for j, p in enumerate(h["docs"]):
            support.append(f"HOP {i+1} PASSAGE {j+1}:\n{p}")
    ctx = "\n\n".join(support)
    
    return (
        "Using the prior sub-questions and their answers along with the supporting context, answer the ORIGINAL QUESTION.\n"
        "The sub-questions have been designed to help you arrive at the final answer step-by-step but may obtain unnecessary details.\n"
        "If the sub-question answers do not provide enough information to answer the original question, you may disregard them and use only the context.\n"
        "The final answer should be concise and directly address the original question, not the sub-questions.\n"
        "If presented with a yes or no question, answer with just 'yes' or 'no'.\n"
        f"ORIGINAL QUESTION:\n{original_question}\n\n"
        f"SUB-QUESTION ANSWERS:\n{mem_lines}\n\n"
        f"SUPPORTING CONTEXT:\n{ctx}\n\n"
        "Final answer (short phrase only):"
    )

LLM_EVAL_SYSTEM_PROMPT = ("""
You are an expert evaluator for a question-answering system.
You are given a question, the gold answer, and the model's answer.
Your job is to judge how semantically correct the model's answer is compared to the gold answer.

The gold answer should be treated as the authoritative reference.
Do not try to correct the gold answer using your own knowledge; only compare the model answer to the gold answer.

Scoring rules:

- Treat an answer as CORRECT if it has the same meaning as the gold answer, even if:
  - It is a sub-span or shorter form that still clearly and uniquely identifies the same entity or fact.
  - It is more specific, as long as the core meaning does not change.
  - It omits non-essential qualifiers such as country names, titles, or extra descriptors, when the main entity or fact is still clear.
- Treat an answer as INCORRECT if it refers to a different entity, missing essential information that changes the meaning or makes the answer significantly less informative for the question, contradicts the gold answer, or is mostly wrong (unknown).
                 
Set:
   - 1.0 for correct answers,
   - 0.0 for incorrect answers.
                 
Important examples:
- Gold: "Canary Islands, Spain"; Model: "Canary Islands" -> correct: score = 1.0
- Gold: "July 4, 1776"; Model: "1776" -> incorrect (missing essential date information): score = 0.0.
- Gold: "Barack Obama"; Model: "Obama" -> correct: score = 1.0.

You must output a single JSON object with the following keys:
- "score": a float either 0 (incorrect) or 1 (correct)
- "explanation": a short string (1-3 sentences)

Do NOT include any other keys.
Do NOT include any text outside of the JSON object.
""")

def build_llm_eval_prompt(question, gold_answer, model_answer):
    return (f"""
        Question:
        {question}

        Gold answer:
        {gold_answer}

        Model answer:
        {model_answer}

        Instructions:
        Use the scoring rules in the system message to decide the score.
        Output only the JSON object.
    """)