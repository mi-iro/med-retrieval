import torch

def get_device_name():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, para):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + para.strip()
    else:
        return title.strip() + ". " + para.strip()