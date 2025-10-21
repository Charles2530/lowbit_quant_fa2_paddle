from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="THUDM/CogVideoX-2b", local_dir="/root/autodl-tmp/CogVideoX-2b"
)
