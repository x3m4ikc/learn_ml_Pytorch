import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def act(x):
    return 0 if x < 0.5 else 1


def go(house, rock, attr):
    X = torch.tensor([house, rock, attr], dtype=torch.float32, device=device)
    Wh = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]], device=device) # matrix 2x3
    Wout = torch.tensor([-1., 1.], device=device)  # vector 1x2

    Zh = torch.mv(Wh, X) # сумма на входах нейронов скрытого слоя
    print("Значения сумм на нейронах скрытого слоя: {}".format(Zh))

    Uh = torch.tensor([act(x) for x in Zh], dtype=torch.float32, device=device)
    print("Значения на выходах нейронов скрытого слоя: {}".format(Uh))

    Zout = torch.dot(Wout, Uh)
    Y = act(Zout)
    print("Выходное значение НС: {}".format(Y))

    return Y


if __name__ == "__main__":
    house = 1
    rock = 1
    attr = 1
    res = go(house, rock, attr)

    print("res: {}".format(res))
    print("I like" if res == 1 else "Dont like")
