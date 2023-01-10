import pickle
import torch
import transformers
import copy

transformers.logging.set_verbosity(50)

class PromptPool(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.total = config.total
        self.prompt_projection = config.prompt_projection
        # if config.
        self.kdim = config.d_model

        self.select_times = torch.ones(6)
        self.update_times = True
        self.select_idx = -1
        self.choose_pool_key = config.choose_pool_key
        # self.device = device

        self.key_list = torch.nn.Parameter(torch.tensor(0.01*torch.rand(self.total, self.kdim), requires_grad=True))
        if self.prompt_projection:
            # # Use a two-layer MLP to encode the prompt
            self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(config.prompt_seq_len, config.hidden_size) for j in range(self.total)])

            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prompt_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prompt_hidden_size, config.num_decoder_layers * 2 * config.hidden_size)
            )
        else:
            self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(config.prompt_seq_len, config.num_decoder_layers * 2 * config.hidden_size)
                               for j in range(self.total)])

    def set_idx(self, idx):
        self.select_idx = idx

    def forward(self, prompt: torch.Tensor):
        if self.prompt_projection:
            if self.choose_pool_key > 0:
                self.select_idx = self.choose_pool_key
            # print(f"select_idx is {self.select_idx}")
            prompt_tokens = self.embeddings[self.select_idx](prompt)
            prompt_kvs = self.trans(prompt_tokens)
        else:
            prompt_kvs = self.embedding[self.select_idx](prompt)
        return prompt_kvs

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