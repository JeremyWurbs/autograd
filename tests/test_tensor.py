import torch
from wichi import Tensor


def test_basic_ops():
    def F_1(x, a, b, c):
        y = (a @ x).T @ (-x @ b).T + c
        return y

    def F_2(x, a, b, c):
        y = (a @ x**2) @ (c.T * x).T / a
        return y

    def F_3(x, a, b, c):
        y = (c.T / x @ -b).sum()
        return y

    def Loss(y_hat, y):
        err = ((y - y_hat) ** 2).sum() / y.numel()
        return err

    x_torch = torch.randn((5, 4), requires_grad=False)
    a_torch = torch.randn((3, 5), requires_grad=True)
    b_torch = torch.randn((4, 3), requires_grad=True)
    c_torch = torch.randn((4, 5), requires_grad=True)

    x = Tensor(x_torch.data, label='x')
    a = Tensor(a_torch.data, label='a')
    b = Tensor(b_torch.data, label='b')
    c = Tensor(c_torch.data, label='c')

    y_1 = F_1(x, a, b, c)
    y_2 = F_2(x, a, b, c)
    y_3 = F_3(x, a, b, c)

    y_torch_1 = F_1(x_torch, a_torch, b_torch, c_torch)
    y_torch_2 = F_2(x_torch, a_torch, b_torch, c_torch)
    y_torch_3 = F_3(x_torch, a_torch, b_torch, c_torch)

    assert torch.allclose(torch.tensor(y_1.data), y_torch_1)
    assert torch.allclose(torch.tensor(y_2.data), y_torch_2)
    assert torch.allclose(torch.tensor(y_3.data), y_torch_3)

    label_1 = torch.randn((4, 5), requires_grad=True)
    label_2 = torch.randn((3, 5), requires_grad=True)
    label_3 = torch.randn((1, 1), requires_grad=True)

    loss_1 = Loss(y_1, label_1.data); loss_1.backward()
    loss_2 = Loss(y_2, label_2.data); loss_2.backward()
    loss_3 = Loss(y_3, label_3.data); loss_3.backward()

    loss_torch_1 = Loss(y_torch_1, label_1.data); loss_torch_1.backward()
    loss_torch_2 = Loss(y_torch_2, label_2.data); loss_torch_2.backward()
    loss_torch_3 = Loss(y_torch_3, label_3.data); loss_torch_3.backward()

    assert torch.allclose(torch.tensor(a.grad), a_torch.grad)
    assert torch.allclose(torch.tensor(b.grad), b_torch.grad)
    assert torch.allclose(torch.tensor(c.grad), c_torch.grad)


def test_advanced_ops():



test_advanced_ops()