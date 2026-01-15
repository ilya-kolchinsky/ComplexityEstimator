import torch
from transformers import AutoTokenizer, AutoModel


def main():
    repo_id = "ilya-kolchinsky/PromptComplexityEstimator"

    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()

    prompt = "Design a distributed consensus protocol with Byzantine fault tolerance..."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_length, padding=True)

    with torch.no_grad():
        score = model(**inputs).logits.squeeze(-1).item()

    print(float(score))


if __name__ == "__main__":
    main()
