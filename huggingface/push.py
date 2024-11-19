from huggingface_hub import HfApi, HfFolder, Repository

repo = Repository(local_dir="models/tinyllama-1.1b", clone_from="tinyllama")
repo.push_to_hub(commit_message="Initial commit")
