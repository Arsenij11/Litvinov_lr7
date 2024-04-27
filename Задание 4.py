import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Генерируем случайные данные для обучения
np.random.seed(42)
torch.manual_seed(42)
X = torch.linspace(0.1, 5, 100).reshape(-1, 1)  # 100 точек в диапазоне от 0.1 до 5
y = 2**X * torch.sin(2**(-X))  # Значения y для каждого x

# Определяем класс нейронной сети
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(1, 40)  # Один входной нейрон, 40 нейронов в скрытом слое
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(40, 1)  # 1 выходной нейрон

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Функция для обучения нейронной сети и вывода графика предсказаний
def train_and_plot():
    model = NeuralNet()  # Создаем модель нейронной сети
    criterion = nn.MSELoss()  # Функция потерь - среднеквадратичное отклонение
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = []  # Список для хранения значений функции ошибки

    # Обучение
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Построение графика предсказаний
    with torch.no_grad():
        predicted = model(X).numpy()

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='b', label='True')
    plt.plot(X, predicted, color='r', label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('True vs Predicted')
    plt.legend()
    plt.show()

    # Возвращаем список значений функции ошибки
    return losses

# Обучение и построение графика
losses = train_and_plot()

# Построение графика функции ошибки
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
