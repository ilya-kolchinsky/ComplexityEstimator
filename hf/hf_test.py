from transformers import AutoTokenizer, AutoModel


def main():
    repo_id = "ilya-kolchinsky/PromptComplexityEstimator"

    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

    x = tok("Write a detailed security audit plan for a kernel module.", return_tensors="pt", truncation=True)
    score = model(**x).logits.squeeze(-1).item()
    print(score)


if __name__ == "__main__":
    main()
