from huggingface_hub import create_repo

repo_url = create_repo("tinyllama", private=False)  # Set private=True for private models
print(f"Model repository created at: {repo_url}")
