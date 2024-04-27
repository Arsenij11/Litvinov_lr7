import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Генерируем случайные данные для обучения
np.random.seed(42)
torch.manual_seed(42)
X = torch.randn(1000, 10)  # 1000 примеров, 10 признаков
y = torch.randint(0, 2, (1000,))  # 1000 меток классов (0 или 1)


# Определяем класс нейронной сети
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# Функция для обучения нейронной сети с заданным lr и записи функции ошибки
def train_and_evaluate(lr):
    model = NeuralNet(10, 40, 1)  # 10 входных признаков, 40 нейронов в скрытом слое, 1 выходной нейрон
    criterion = nn.BCELoss()  # бинарная кросс-энтропия для бинарной классификации
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []  # Список для хранения значений функции ошибки

    # Обучение
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y.float())  # squeeze используется для удаления размерности 1
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # Записываем значение функции ошибки

    # Возвращаем список значений функции ошибки
    return losses


# Номинальное значение lr
nominal_lr = 0.01

# Интервал для тестирования lr
lr_values = [0.1 * nominal_lr, 0.5 * nominal_lr, nominal_lr, 2 * nominal_lr, 5 * nominal_lr, 10 * nominal_lr]

# Исследование нейронной сети с разными значениями lr и построение графика функции ошибки
for lr in lr_values:
    losses = train_and_evaluate(lr)
    plt.plot(range(1, len(losses) + 1), losses, label=f"lr={lr}")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss with Different Learning Rates')
plt.legend()
plt.show()
