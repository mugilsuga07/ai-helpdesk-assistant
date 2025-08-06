from datasets import load_dataset

dataset = load_dataset("daily_dialog", trust_remote_code=True)

with open("data/sample.txt", "w") as f:
    for item in dataset["train"]:
        dialog = "\n".join(item["dialog"])
        f.write(dialog + "\n\n")

print("Dialog data saved to data/sample.txt")

