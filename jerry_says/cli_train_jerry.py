import argparse
import os
import time

import torch

from jerry_says.data_reader import (
    SimpleIterator, build_dataset_and_vocab, get_sentences_by_jerry
)
from jerry_says.model import (
    RNN_SIZE, build_transformer_model, count_parameters
)
from jerry_says.trainer import (
     ModifiedAdamOptimizer, train_epoch, validate_epoch
)


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-epochs', '--epochs', type=int, default=600, help='Choose the number of epochs.'
    )
    ap.add_argument(
        '-batch_size', '--batch_size', type=int, default=200, help='Batch size.'
    )

    return vars(ap.parse_args())


def _main():
    args = argument_parser()
    num_epochs = int(args['epochs'])
    batch_size = int(args['batch_size'])

    print()
    if torch.cuda.is_available():
        device = 'cuda'
        print('Training on GPU.')
    else:
        device = 'cpu'
        print('Training on CPU.')

    sentences_by_jerry = get_sentences_by_jerry()
    train_dataset, valid_dataset, src_field, tgt_field = \
        build_dataset_and_vocab(sentences_by_jerry)

    src_vocab_size = len(src_field.vocab)
    tgt_vocab_size = len(tgt_field.vocab)
    print(f'Vocabulary size of source is {src_vocab_size:d}.')
    print(f'Vocabulary size of target is {tgt_vocab_size:d}.')

    src_pad_index = src_field.vocab.stoi[src_field.pad_token]
    tgt_pad_index = tgt_field.vocab.stoi[tgt_field.pad_token]

    train_iterator = SimpleIterator(
        dataset=train_dataset, batch_size=batch_size, device=device, train=True
    )
    valid_iterator = SimpleIterator(
        dataset=valid_dataset, batch_size=batch_size, device=device, train=False
    )

    model = build_transformer_model(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size
    )
    count_parameters(model)

    # Initialize parameter values: xavier_uniform_() fills the input Tensor with values
    # using a uniform distribution which depends on the number of input and output units
    # in the weight tensor).
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    model.to(device)

    optimizer = ModifiedAdamOptimizer(model, RNN_SIZE)
    criterion = torch.nn.NLLLoss(ignore_index=tgt_pad_index, reduction='sum')

    save_dir = 'trained_model'
    if not os.path.isdir(f'{save_dir}'):
        os.makedirs(f'{save_dir}')
    model_save_path = os.path.join(save_dir, 'transformer-for-jerry.pt')

    best_valid_loss = float('inf')
    start_time = time.time()

    print(f'Batch size is {batch_size}.')
    print(
        f'Training on {len(train_iterator)} batches and '
        f'validating on {len(valid_iterator)} batches.\n'
    )
    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_lr = train_epoch(
            model, train_iterator, optimizer, criterion, src_pad_index, tgt_pad_index
        )
        valid_loss, valid_accuracy = validate_epoch(
            model, valid_iterator, criterion, src_pad_index, tgt_pad_index
        )

        if valid_loss < best_valid_loss:
            print(
                f'Valid loss {valid_loss:.3f} is less than best valid loss '
                f'{best_valid_loss:.3f}!'
            )
            best_valid_loss = valid_loss

            checkpoint = {
                'model': model.state_dict(),
                'src_field': src_field,
                'tgt_field': tgt_field
            }
            print(f'Saving checkpoint {model_save_path:s}')
            torch.save(checkpoint, model_save_path)
        else:
            print('Validation loss did not improve!')

        elapsed_time = time.time() - start_time
        print(f'Epoch: {epoch + 1} (of {num_epochs}); elapsed time: {elapsed_time / 60:.1f}m')
        print(f'Train loss: {train_loss:.3f}; accuracy: {train_accuracy:.3f}')
        print(f'Valid loss: {valid_loss:.3f}; accuracy: {valid_accuracy:.3f}')
        print(f'Train lr: {train_lr:.3E}\n')


def main():
    """
    The setuptools entry point.
    """
    _main()
