from transformers import VisionEncoderDecoderModel, AutoTokenizer
import torch

# Load the full VisionEncoderDecoderModel and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("Matthijs0/im2latex_base")
tokenizer = AutoTokenizer.from_pretrained("Matthijs0/im2latex_base")

# Input string to feed to the decoder
input_string = "f(x)="
input_ids = tokenizer.encode(input_string, return_tensors="pt")

# Simulate encoder outputs (dummy embeddings)
dummy_encoder_outputs = torch.randn(1, 1, model.config.decoder.hidden_size)  # (batch_size, seq_length, hidden_size)

# Generate text using the decoder
decoder_output = model.decoder.generate(
    input_ids=input_ids,
    encoder_hidden_states=dummy_encoder_outputs,  # Fake encoder outputs
    max_length=100,
)

# Decode the output tokens into text
generated_texts = tokenizer.batch_decode(decoder_output, skip_special_tokens=True)

print("Generated LaTeX formula:", generated_texts[0])
