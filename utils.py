import torch
from tqdm.auto import tqdm

from PhoBERT_base import BertBase
from PhoBERT_LSTM import BertLSTM


def get_model(task):
    # 1 for topic, 2 for sentiment, 3 for status
    if task == 1:
        model = BertLSTM(n_classes=4)
    elif task == 2:
        model = BertBase(n_classes=3)
    else:
        model = BertLSTM(n_classes=1)

    return model


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Load model from {path}")


def predict(model, input_ids, attention_masks, device):
    model.eval()
    predictions = []
    prob = []

    input_ids.to(device)
    attention_masks.to(device)

    output = model(input_ids, attention_masks)

    _, pred = torch.max(output, 1)
    # predictions.append(preds.flatten())

    numpy_logits = output.cpu().detach().numpy()
    prob += list(numpy_logits)

    # predictions = torch.cat(predictions).detach().cpu()

    return pred[0], prob[0][0]

