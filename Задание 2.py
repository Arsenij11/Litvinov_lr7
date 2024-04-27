import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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


# Функция для обучения и оценки нейронной сети с заданным lr
def train_and_evaluate(lr):
    model = NeuralNet(10, 40, 1)  # 10 входных признаков, 40 нейронов в скрытом слое, 1 выходной нейрон
    criterion = nn.BCELoss()  # бинарная кросс-энтропия для бинарной классификации
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Обучение
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y.float())  # squeeze используется для удаления размерности 1
        loss.backward()
        optimizer.step()

    # Оценка точности модели
    with torch.no_grad():
        predicted = (model(X).squeeze() > 0.5).float()
        accuracy = (predicted == y).sum().item() / y.size(0)

    return accuracy


# Номинальное значение lr
nominal_lr = 0.01

# Интервал для тестирования lr
lr_values = [0.1 * nominal_lr, 0.5 * nominal_lr, nominal_lr, 2 * nominal_lr, 5 * nominal_lr, 10 * nominal_lr]

# Исследование нейронной сети с разными значениями lr
for lr in lr_values:
    accuracy = train_and_evaluate(lr)
    with open("Задание_2.txt", "at") as f:
        f.write(f"Learning Rate: {lr}, Accuracy: {accuracy}\n")

