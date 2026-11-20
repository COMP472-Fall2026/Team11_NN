import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)

def create_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=64):
  train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
  val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, val_loader

class SimpleMLP(nn.Module):
  def __init__(self, input_size, output_size):
    super(SimpleMLP, self).__init__()
    self.network = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_size)
    )
  def forward(self, x):
    return self.network(x)

def train_and_evaluate(model, train_loader, val_loader, epochs=15, title="Neural Network"):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  train_losses = []
  val_accuracies = []

  print(f"\n=====================================")
  print(f"--- Training {title} ---")

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_acc = correct / total
    val_accuracies.append(val_acc)

    if (epoch +1) % 5 == 0 or epoch == 0:
      print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f} | Validation Accuracy: {val_acc: .2%}")

  print(f"Final Validation Accuracy for {title}: {val_accuracies[-1]:.2%}")

  fig, ax1 = plt.subplots(figsize=(7,4))

  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Training Loss', color='red')
  ax1.plot(range(1, epochs+1), train_losses, color='red', marker='o', label='Train Loss')
  ax1.tick_params(axis='y', labelcolor='red')

  ax2 = ax1.twinx()
  ax2.set_ylabel('Validation Accuracy', color='blue')
  ax2.plot(range(1, epochs+1), val_accuracies, color='blue', marker='s', label='Val Accuracy')
  ax2.tick_params(axis='y', labelcolor='blue')

  plt.title(f'{title} - Learning Curve')
  fig.tight_layout()
  plt.grid(alpha=0.3)
  plt.show()

  return model