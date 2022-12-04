import pickle
import torch
import transformers
import copy

transformers.logging.set_verbosity(50)

class PromptPool(torch.nn.Module):
    def __init__(self, config, device):
        # self.total = config.total
        # self.prompt_projection = config.prompt_projection
        # self.kdim = 768
        # self.key_list = []
        # self.device = device
        # for i in self.total:
        #     self.key_list.append([torch.tensor(0.01*torch.rand(self.kdim),requires_grad=True,device=device)])
        #
        # if self.prompt_projection:
        #     # Use a two-layer MLP to encode the prompt
        #     self.embeddings = [torch.nn.Embedding(config.prompt_seq_len, config.hidden_size) for j in self.total]
        #     self.embedding = torch.nn.Embedding(config.prompt_seq_len, config.hidden_size)
        #     self.trans = torch.nn.Sequential(
        #         torch.nn.Linear(config.hidden_size, config.prompt_hidden_size),
        #         torch.nn.Tanh(),
        #         torch.nn.Linear(config.prompt_hidden_size, config.num_decoder_layers * 2 * config.hidden_size)
        #     )
        # else:
        #     self.embeddings = [torch.nn.Embedding(config.prompt_seq_len, config.num_decoder_layers * 2 * config.hidden_size)
        #                        for j in self.total]
        pass
    def loadPool(self):
        pass

    def savePool(self):
        pass


class PromptEncoder(torch.nn.Module):
    r'''
        The torch.nn model to encode the prompt

        Input shape: (batch-size, prompt-length)

        Output shape: (batch-size, prompt-length, 2*layers*hidden)
        '''

    def __init__(self, config):
        super().__init__()
        self.prompt_projection = config.prompt_projection
        if self.prompt_projection:
            # Use a two-layer MLP to encode the prompt
            self.embedding = torch.nn.Embedding(config.prompt_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prompt_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prompt_hidden_size, config.num_decoder_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.prompt_seq_len, config.num_decoder_layers * 2 * config.hidden_size)

    def forward(self, prompt: torch.Tensor):
        if self.prompt_projection:
            prompt_tokens = self.embedding(prompt)
            prompt_kvs = self.trans(prompt_tokens)
        else:
            prompt_kvs = self.embedding(prompt)
        return prompt_kvs