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
    x_1_wichi = Tensor([[2., 1.5], [-1.5, 3.]])
    x_1_torch = torch.Tensor(x_1_wichi.data); x_1_torch.requires_grad = True

    x_2_wichi = x_1_wichi.relu()
    x_2_torch = torch.relu(x_1_torch)
    assert torch.allclose(torch.tensor(x_2_wichi.data), x_2_torch)

    x_3_wichi = x_1_wichi.tanh()
    x_3_torch = x_1_torch.tanh()
    assert torch.allclose(torch.tensor(x_3_wichi.data), x_3_torch)

    x_4_wichi = x_3_wichi.exp()
    x_4_torch = x_3_torch.exp()
    assert torch.allclose(torch.tensor(x_4_wichi.data), x_4_torch)

    x_5_wichi = x_4_wichi.reshape((4, 1))
    x_5_torch = x_4_torch.reshape((4, 1))
    assert torch.allclose(torch.tensor(x_5_wichi.data), x_5_torch)

    x_6_wichi = x_5_wichi.sum()
    x_6_torch = x_5_torch.sum()
    assert torch.allclose(torch.tensor(x_6_wichi.data), x_6_torch)

    x_6_wichi.backward()
    x_6_torch.backward()
    assert torch.allclose(torch.tensor(x_1_wichi.grad), x_1_torch.grad)
