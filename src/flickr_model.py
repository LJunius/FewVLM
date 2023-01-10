import torch
import torch.nn as nn


from modeling_t5 import FewVLM


class FewVLMCOCOCaption(FewVLM):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result

    def infer_step(self, input_ids, vis_feats, vis_pos, **kwargs):
        device = next(self.parameters()).device
        vis_feats = vis_feats.to(device)
        input_ids = input_ids.to(device)
        vis_pos = vis_pos.to(device)
        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return generated_sents