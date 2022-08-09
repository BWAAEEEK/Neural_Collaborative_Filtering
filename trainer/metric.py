import torch

def accuracy(output, target):
    with torch.no_grad():
        output = output.view(len(output))
        target = target.view(len(target))
        pred = (output > 0.5)

        correct = torch.sum(pred == target)

    return correct / len(target)