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
        optimizer = torch.optim.SGD(sine_net.parameters(), lr)
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
    plt.savefig('Задание_1.png')

MAE_var8 = test_sine_net('SGD', 40, 0.01)
print(f'Метрика = {MAE_var8}')
