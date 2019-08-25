import math

import torch
from torch.nn import Module
from tqdm import tqdm

from jerry_says.model import (
    generate_key_padding_masks, generate_square_subsequent_mask
)


def train_epoch(model, iterator, optimizer, criterion, generator, pad_index):
    model.train()

    epoch_loss = 0
    report_every = math.ceil(len(iterator) / 25)
    print(f'Total training batches {len(iterator)}')
    for i, batch in enumerate(tqdm(iterator)):
        optimizer.optimizer.zero_grad()

        src = batch.src
        tgt = batch.tgt

        # exclude last target
        tgt = tgt[:-1]

        tgt_mask = generate_square_subsequent_mask(tgt.shape[0])

        src_key_padding_mask, tgt_key_padding_mask = generate_key_padding_masks(
            src=src, tgt=tgt, pad_index=pad_index
        )
        output = model(
            src=src, tgt=tgt, src_mask=None, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        # Output is of size = T x N x E; reshape it to (T x N) x E
        output_reshaped = output.view(-1, output.size(2))
        scores = generator(output_reshaped)
        # Target is of size T x N; reshape it to (T x N)
        target_reshaped = tgt.view(-1)

        # Calculate loss
        loss = criterion(scores, target_reshaped) / float(batch.batch_size)

        loss.backward()
        optimizer.step()

        # Accuracy calculation
        prediction = scores.max(1)[1]
        non_padding = target_reshaped.ne(pad_index)
        num_correct = prediction.eq(
            target_reshaped
        ).masked_select(
            non_padding
        ).sum().item()
        num_non_padding = non_padding.sum().item()
        accuracy = 100.0 * num_correct / num_non_padding

        if i % report_every == 0:
            tqdm.write(
                f'Step: {i}; '
                f'training loss: {loss.item():.3f}; '
                f'accuracy: {accuracy:.1f} '
                f'lr: {optimizer._rate:.3E}'
            )

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def validate_epoch(model, iterator, criterion, generator, pad_index):
    model.eval()

    epoch_loss = 0
    report_every = math.ceil(len(iterator) / 4)

    print(f'Total validation batches {len(iterator)}')
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch.src
            tgt = batch.tgt

            # exclude last target
            tgt = tgt[:-1]

            tgt_mask = generate_square_subsequent_mask(tgt.shape[0])

            src_key_padding_mask, tgt_key_padding_mask = generate_key_padding_masks(
                src=src, tgt=tgt, pad_index=pad_index
            )
            output = model(
                src=src, tgt=tgt, src_mask=None, tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            # Output is of size = T x N x E; reshape it to (T x N) x E
            output_reshaped = output.view(-1, output.size(2))
            scores = generator(output_reshaped)
            # Target is of size T x N; reshape it to (T x N)
            target_reshaped = tgt.view(-1)

            # Calculate loss
            loss = criterion(scores, target_reshaped) / float(batch.batch_size)

            # Accuracy calculation
            prediction = scores.max(1)[1]
            non_padding = target_reshaped.ne(pad_index)
            num_correct = prediction.eq(
                target_reshaped
            ).masked_select(
                non_padding
            ).sum().item()
            num_non_padding = non_padding.sum().item()
            accuracy = 100.0 * num_correct / num_non_padding

            if i % report_every == 0:
                tqdm.write(
                    f'Step: {i}; '
                    f'validation loss: {loss.item():.3f}; '
                    f'accuracy: {accuracy:.1f}'
                )

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


class LabelSmoothingLoss(Module):
    """
    Copied from OpenNMT
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return torch.nn.functional.kl_div(output, model_prob, reduction='sum')


class ModifiedAdamOptimizer:
    def __init__(self, model, rnn_size):
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9
        )
        self.rnn_size = rnn_size
        self.warmup = 20
        self.factor = 0.1
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
