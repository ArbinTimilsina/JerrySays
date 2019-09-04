import torch
from tqdm import tqdm

from jerry_says.model import (
    generate_key_padding_masks, generate_square_subsequent_mask
)


def train_epoch(model, iterator, optimizer, criterion, src_pad_index, tgt_pad_index):
    model.train()

    epoch_loss = 0
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

        clip = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator), accuracy, optimizer._rate


def validate_epoch(model, iterator, criterion, src_pad_index, tgt_pad_index):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
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
            epoch_loss += loss.item()

    return epoch_loss / len(iterator), accuracy


class ModifiedAdamOptimizer:
    """
    Adam optimizer such that learning rate increases linearly for warmup steps and
    decreases thereafter proportionally to the inverse square root of the step number.
    """
    def __init__(self, model, rnn_size):
        self.optimizer = torch.optim.Adam(model.parameters())
        self.rnn_size = rnn_size
        self.warmup = 53
        self.factor = 0.01
        self._step = 0
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
                self.rnn_size ** (-0.5) * min(
            step ** (-0.5), step * self.warmup ** (-1.5))
        )
