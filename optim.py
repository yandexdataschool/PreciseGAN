from torch.optim.adam import Adam
from torch.optim.sgd import SGD


def setup_optimizer(model, lr, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = Adam(optimizer_grouped_parameters, lr=lr)
    # optimizer = SGD(optimizer_grouped_parameters, lr=lr)

    return optimizer
