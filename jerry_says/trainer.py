import math

import torch
from tqdm import tqdm

from jerry_says.model import (
    generate_key_padding_masks, generate_square_subsequent_mask
)


def train_epoch(model, iterator, optimizer, criterion, src_pad_index, tgt_pad_index):
    model.train()

    epoch_loss = 0
    report_every = math.ceil(len(iterator) / 25)
    print(f'Total training batches {len(iterator)}')
    for i, batch in enumerate(tqdm(iterator)):
        optimizer.optimizer.zero_grad()

        src = batch.src
        tgt = batch.tgt

        # Exclude last target from input
        tgt_input = tgt[:-1]
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[0])

        src_key_padding_mask, tgt_key_padding_mask = generate_key_padding_masks(
            src=src, tgt=tgt_input, src_pad_index=src_pad_index, tgt_pad_index=tgt_pad_index
        )
        output = model(
            src=src, tgt=tgt_input, src_mask=None, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        # Output is of size = T x N x E; reshape it to (T x N) x E
        output_reshaped = output.view(-1, output.size(2))
        scores = model.generator(output_reshaped)

        # Target is of size T x N; reshape it to (T x N)
        tgt_for_output = tgt[1:]
        target_reshaped = tgt_for_output.view(-1)

        # Calculate accuracy
        prediction = torch.argmax(scores, 1)
        non_padding = target_reshaped.ne(tgt_pad_index)
        num_correct = prediction.eq(
            target_reshaped
        ).masked_select(
            non_padding
        ).sum().item()
        num_non_padding = non_padding.sum().item()
        accuracy = 100.0 * num_correct / num_non_padding

        # Calculate loss and normalize by no. of tokens
        loss = criterion(scores, target_reshaped) / float(num_non_padding)
        loss.backward()
        optimizer.step()

        if i % report_every == 0:
            tqdm.write(
                f'Step: {i+1}; '
                f'training loss: {loss.item():.3f}; '
                f'accuracy: {accuracy:.1f} '
                f'lr: {optimizer._rate:.3E}'
            )

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def validate_epoch(model, iterator, criterion, src_pad_index, tgt_pad_index):
    model.eval()

    epoch_loss = 0

    print(f'Total validation batches {len(iterator)}')
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch.src
            tgt = batch.tgt

            # Exclude last target from input
            tgt_input = tgt[:-1]
            tgt_mask = generate_square_subsequent_mask(tgt_input.shape[0])

            src_key_padding_mask, tgt_key_padding_mask = generate_key_padding_masks(
                src=src, tgt=tgt_input, src_pad_index=src_pad_index,
                tgt_pad_index=tgt_pad_index
            )
            output = model(
                src=src, tgt=tgt_input, src_mask=None, tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            # Output is of size = T x N x E; reshape it to (T x N) x E
            output_reshaped = output.view(-1, output.size(2))
            scores = model.generator(output_reshaped)

            # Target is of size T x N; reshape it to (T x N)
            tgt_for_output = tgt[1:]
            target_reshaped = tgt_for_output.view(-1)

            # Calculate accuracy
            prediction = torch.argmax(scores, 1)
            non_padding = target_reshaped.ne(tgt_pad_index)
            num_correct = prediction.eq(
                target_reshaped
            ).masked_select(
                non_padding
            ).sum().item()
            num_non_padding = non_padding.sum().item()
            accuracy = 100.0 * num_correct / num_non_padding

            # Calculate loss and normalize by no. of tokens
            loss = criterion(scores, target_reshaped) / float(num_non_padding)

            if i == len(iterator) - 1:
                tqdm.write(
                    f'Step: {i+1}; '
                    f'validation loss: {loss.item():.3f}; '
                    f'accuracy: {accuracy:.1f}'
                )

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


class ModifiedAdamOptimizer:
    """
    Adam optimizer such that learning rate increases linearly for warmup steps and
    decreases thereafter proportionally to the inverse square root of the step number.
    """
    def __init__(self, model):
        self.optimizer = torch.optim.Adam(model.parameters())
        self.factor = 1e-5
        self.start_step = 21
        self._step = 0
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate(self._step)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step):
        if step >= self.start_step:
            step = step - self.start_step + 1
        else:
            step = 1.0
        return self.factor * step ** (-0.01)
