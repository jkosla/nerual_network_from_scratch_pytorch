import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from dataset import XORDataset, visualize_classification
from simple_nn import SimpleClassifier, GradientDescent


def train_model(model, data_loader, optimizer, num_epochs=100, device="cpu"):
    for epoch in range(num_epochs):
        for data_inputs, data_labels in data_loader:
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

            preds = torch.sigmoid(model.forward(data_inputs).squeeze(dim=1))
            dL_dW1, dL_db1, dL_dW2, dL_db2 = model.backprop(
                data_inputs, data_labels, preds
            )

            optimizer.step(model.linear1, dL_dW1, dL_db1)
            optimizer.step(model.linear2, dL_dW2, dL_db2)


def eval_model(model, data_loader, device="cpu"):
    correct, total = 0, 0

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = torch.sigmoid(model.forward(data_inputs).squeeze(dim=1))
            pred_labels = (preds >= 0.5).long()

            correct += (pred_labels == data_labels).sum().item()
            total += data_labels.size(0)

    print(f"Model Accuracy: {100.0 * correct / total:.2f}%")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = XORDataset(size=2500)
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = XORDataset(size=500)
    test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1, device=device)
    optimizer = GradientDescent(lr=0.01)

    visualize_classification(model, test_dataset.data, test_dataset.label)

    train_model(model, train_loader, optimizer, device=device)
    eval_model(model, test_loader, device=device)

    visualize_classification(model, test_dataset.data, test_dataset.label)
    plt.show()
