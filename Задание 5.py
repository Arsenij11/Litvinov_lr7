import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Генерируем случайные данные для обучения
np.random.seed(42)
torch.manual_seed(42)
X_train = torch.linspace(0.1, 5, 100).reshape(-1, 1)  # 100 точек в диапазоне от 0.1 до 5 для обучения
X_validation = torch.linspace(0.1, 5, 50).reshape(-1, 1)  # 50 точек в диапазоне от 0.1 до 5 для валидации
y_train = 2**X_train * torch.sin(2**(-X_train))  # Значения y для каждого x для обучения
y_validation = 2**X_validation * torch.sin(2**(-X_validation))  # Значения y для каждого x для валидации

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

def train_model(architecture, loss_func, lr, num_epochs):
    # Создаем модель
    model = NeuralNet()

    # Оптимизатор
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Функция потерь
    criterion = loss_func

    # Обучение модели
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Вычисляем MAE на валидационном наборе данных
    with torch.no_grad():
        predicted = model(X_validation)
        mae = torch.mean(torch.abs(predicted - y_validation)).item()

    return mae, losses

# Варьируемые параметры
architectures = [(1, 40), (2, 40, 20), (3, 50, 30, 20)]  # Разные архитектуры сети
loss_functions = [nn.MSELoss(), nn.L1Loss()]  # Разные loss-функции
learning_rates = [0.001, 0.01, 0.1]  # Разные learning rates
num_epochs_list = [100, 500, 1000]  # Разное количество эпох в обучении

best_mae = float('inf')
best_params = None

# Перебираем все комбинации параметров
for architecture in architectures:
    for loss_func in loss_functions:
        for lr in learning_rates:
            for num_epochs in num_epochs_list:
                mae, _ = train_model(architecture, loss_func, lr, num_epochs)
                if mae < 0.03 and mae < best_mae:
                    best_mae = mae
                    best_params = {
                        'architecture': architecture,
                        'loss_func': loss_func,
                        'lr': lr,
                        'num_epochs': num_epochs
                    }

print("Best MAE:", best_mae)
print("Best Parameters:", best_params)
