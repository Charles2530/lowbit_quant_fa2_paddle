from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zai-org/CogVideoX-2b", local_dir="./models/CogVideoX-2b"
)
