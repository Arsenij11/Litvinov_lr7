import torch
import matplotlib.pyplot as plt
import matplotlib

# Устанавливаем параметры графика
matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)

# Генерируем обучающие данные
x_train = torch.rand(100) * 20.0 - 10.0
y_train = torch.sin(x_train)
noise = torch.randn(y_train.shape) / 5.0
y_train += noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

# Генерируем валидационные данные
x_validation = torch.linspace(-10, 10, 100)
y_validation = torch.sin(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

# Определяем класс нейронной сети
class SineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


# Определяем функцию оценки метрики
def metric(pred, target):
    return (pred - target).abs().mean()


# Определяем функцию потерь
def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()

# Определяем функцию обучения
def train(sine_net, optimizer, epoch_count):
    for epoch_index in range(epoch_count):
        optimizer.zero_grad()
        y_pred = sine_net.forward(x_train)
        loss_val = loss(y_pred, y_train)
        loss_val.backward()
        optimizer.step()

# Определяем функцию тестирования нейронной сети
def test_sine_net(opt_method, hidden_neurons, lr):
    sine_net = SineNet(hidden_neurons)
    if opt_method == 'ADAM':
        optimizer = torch.optim.Adam(sine_net.parameters(), lr)
    elif opt_method == 'SGD':
        optimizer = torch.optim.SGD(sine_net.parameters(), lr=lr)  # Изменено: передан параметр lr
    else:
        return
    train(sine_net, optimizer, 2000)
    predict(sine_net, x_validation.detach(), y_validation.detach())
    print(opt_method, hidden_neurons, lr, 'finished')
    return metric(sine_net.forward(x_validation), y_validation).item()


# Определяем функцию предсказания
def predict(net, x, y):
    y_pred = net.forward(x)
    plt.plot(x.numpy(), y.numpy(), 'o', label='Ground truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig('Задание_2.png')

result_MAE = 1
result_check_num = 1
check_num = 1
lr = 0.01  # Шаг градиентного спуска
n_hidden_neurons = 40  # Число нейронов в скрытом слое

while lr <= 0.02:  # Изменено: условие для шага градиентного спуска
    print('Тест №', check_num)
    MAE = test_sine_net('SGD', n_hidden_neurons, lr)  # Изменено: переданы параметры n_hidden_neurons и lr
    if MAE < result_MAE:
        result_MAE = MAE
        result_check_num = check_num
    print('Метрика =', MAE)
    lr += 0.001  # Изменено: увеличивается шаг градиентного спуска
    check_num += 1
    print()

print('Лучшая метрика', result_MAE)
print('Номер исследования', result_check_num)
