import torch

from jerry_says.data_reader import tokenize_en
from jerry_says.model import build_transformer_model


def build_model_to_serve(saved_model_path):
    """
    Build the transformer model, load the trained parameters, and put it in the
    evaluation mode.
    """
    checkpoint = torch.load(saved_model_path)

    src_field = checkpoint['src_field']
    tgt_field = checkpoint['tgt_field']
    src_vocab_size = len(src_field.vocab)
    tgt_vocab_size = len(tgt_field.vocab)

    model = build_transformer_model(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size
    )
    # Load the trained parameters
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, src_field, tgt_field


def greedy_server(saved_model_path, incomplete_sentence, max_length=20):
    """
    Predict the next word using greedy decoding.
    """
    model, src_field, tgt_field = build_model_to_serve(saved_model_path)

    tokenized_sentence = tokenize_en(incomplete_sentence)
    tokenized_sentence = [t.lower() for t in tokenized_sentence]
    numericalized_sentence = [src_field.vocab.stoi[t] for t in tokenized_sentence]
    seed_tensor = torch.LongTensor(numericalized_sentence).unsqueeze(1)

    memory = model.encode(src=seed_tensor, src_mask=None, src_key_padding_mask=None)

    tgt_tensor = torch.ones(1, 1).fill_(
        tgt_field.vocab.stoi[tgt_field.init_token]
    ).type_as(seed_tensor.data)

    complete_sentence = incomplete_sentence
    for i in range(max_length):
        output = model.decode(
            memory=memory, tgt=tgt_tensor, tgt_mask=None, tgt_key_padding_mask=None
        )
        output_reshaped = output.view(-1, output.size(2))
        scores = model.generator(output_reshaped)
        next_word_tensor = torch.argmax(scores, 1)
        next_word = next_word_tensor.data[0]

        tgt_tensor = torch.cat(
            [tgt_tensor, torch.ones(1, 1).type_as(seed_tensor.data).fill_(next_word)],
            dim=0
        )
        if tgt_field.vocab.itos[next_word.item()] == tgt_field.eos_token:
            break
        complete_sentence += ' ' + tgt_field.vocab.itos[next_word.item()]

    return complete_sentence
