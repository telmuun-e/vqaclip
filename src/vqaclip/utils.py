import torch
import torch.nn.functional as nnf


def generate(
        model,
        tokenizer,
        prompt,
        embed,
        clip_length,
        entry_count=1,
        entry_length=15,
        top_p=0.8,
        temperature=1.,
        device="cuda"
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.eos_token_id
    filter_value = -float("Inf")
    embed = model.mapping_network(embed).reshape(1, clip_length, -1)

    with torch.no_grad():

        for entry_idx in range(entry_count):
            tokens = torch.tensor(tokenizer.encode(prompt))
            q_len = len(tokens)
            tokens = tokens.unsqueeze(0).to(device)

            generated = model.gpt.transformer.wte(tokens)
            generated = torch.cat([embed, generated], 1)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list[q_len:-1])
            generated_list.append(output_text)

    return generated_list[0]