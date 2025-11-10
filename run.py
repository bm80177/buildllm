import torch
from src.pre_processing import create_dataloader_v1, tokenizer, create_embedding_layers

# ===============================================
# Example Usage
# ===============================================
if __name__ == "__main__":
    # ----------------------------
    # Load raw text
    # ----------------------------
    with open("./_data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # ----------------------------
    # Parameters
    # ----------------------------
    batch_size = 2
    max_length = 4
    stride = 1

    # ----------------------------
    # Create dataloader
    # ----------------------------
    dataloader, dataset = create_dataloader_v1(
        raw_text, batch_size=batch_size, max_length=max_length, stride=stride, shuffle=False
    )

    # ----------------------------
    # Inspect first two batches
    # ----------------------------
    data_iter = iter(dataloader)
    for batch_num in range(2):
        input_ids, target_ids = next(data_iter)
        print(f"Batch {batch_num + 1}:")
        for i in range(input_ids.shape[0]):
            print("Decoded Input :", tokenizer.decode(input_ids[i].tolist()))
            print("Decoded Target:", tokenizer.decode(target_ids[i].tolist()))
        print("\n")

    # ----------------------------
    # Create embeddings
    # ----------------------------
    vocab_size = tokenizer.n_vocab
    token_emb, pos_emb = create_embedding_layers(vocab_size, max_length)

    # Example: combine token and positional embeddings for a batch
    input_embeddings = token_emb(input_ids) + pos_emb(torch.arange(max_length))
    print("Input Embeddings Shape:", input_embeddings.shape)

    # ===============================================
    # Generate Complete Encoded Dataset for GPT
    # ===============================================
    all_input_ids = torch.stack(dataset.input_ids)   # Shape: [num_sequences, max_length]
    all_target_ids = torch.stack(dataset.target_ids) # Shape: [num_sequences, max_length]

    print("Complete Dataset Shapes:")
    print("Input IDs:", all_input_ids.shape)
    print("Target IDs:", all_target_ids.shape)

    # Optional: Save for training
    torch.save({
        "input_ids": all_input_ids,
        "target_ids": all_target_ids
    }, "./outputs/gpt_training_data.pt")
    
    print("Complete encoded dataset saved to 'gpt_training_data.pt'.")
