import pytest
import torch
from wichi import Value


def test_basic_ops():
    def F_1(x, a, b, c):
        y = (a * x) + (-x * b) + c
        return y

    def F_2(x, a, b, c):
        y = (a * x**2) + (b * x) + c
        return y

    def F_3(x, a, b, c):
        y = (a / x**-0.5) + (-x / b) + 1/c
        return y

    def Loss(y_hat, y=5.):
        err = (y - y_hat) ** 2
        return err

    x = Value(3.)
    a = Value(2.)
    b = Value(-4.)
    c = Value(-2.5)

    x_torch = torch.Tensor([x.data])
    a_torch = torch.Tensor([a.data]); a_torch.requires_grad = True
    b_torch = torch.Tensor([b.data]); b_torch.requires_grad = True
    c_torch = torch.Tensor([c.data]); c_torch.requires_grad = True

    y_1 = F_1(x, a, b, c)
    y_2 = F_2(x, a, b, c)
    y_3 = F_3(x, a, b, c)

    y_torch_1 = F_1(x_torch, a_torch, b_torch, c_torch)
    y_torch_2 = F_2(x_torch, a_torch, b_torch, c_torch)
    y_torch_3 = F_3(x_torch, a_torch, b_torch, c_torch)

    assert y_1.data == pytest.approx(y_torch_1.data.item(), 1e-3)
    assert y_2.data == pytest.approx(y_torch_2.data.item(), 1e-3)
    assert y_3.data == pytest.approx(y_torch_3.data.item(), 1e-3)

    print(f'\nx: {x}, a: {a}, b: {b}, c: {c}')
    print(f'x_torch: {x_torch}, a_torch: {a_torch}, b_torch: {b_torch}, c_torch: {c_torch}')
    print(f'y_1: {y_1}, y_2: {y_2}, y_3: {y_3}')

    loss_1 = Loss(y_1); loss_1.backward()
    loss_2 = Loss(y_2); loss_2.backward()
    loss_3 = Loss(y_3); loss_3.backward()

    loss_torch_1 = Loss(y_torch_1); loss_torch_1.backward()
    loss_torch_2 = Loss(y_torch_2); loss_torch_2.backward()
    loss_torch_3 = Loss(y_torch_3); loss_torch_3.backward()

    assert a.grad == pytest.approx(a_torch.grad.item(), 1e-3)
    assert b.grad == pytest.approx(b_torch.grad.item(), 1e-3)
    assert c.grad == pytest.approx(c_torch.grad.item(), 1e-3)

    print(f'loss_1: {loss_1.data}, loss_torch_1: {loss_torch_1}')
    print(f'a.grad: {a.grad}, b.grad: {b.grad}, c.grad: {c.grad}')
    print(f'a_torch.grad: {a_torch.grad.item()}, b_torch.grad: {b_torch.grad.item()}, c_torch.grad: {c_torch.grad.item()}')


def test_advanced_ops():
    x_1 = Value(1.5)
    x_torch_1 = torch.Tensor([1.5]); x_torch_1.requires_grad = True

    x_2 = x_1.relu()
    x_torch_2 = torch.relu(x_torch_1)
    assert x_2.data == pytest.approx(x_torch_2.data.item(), 1e-3)
    print(f'\nx: {x_2.data}, x_torch: {x_torch_2.data.item()}')

    x_3 = x_2.tanh()
    x_torch_3 = torch.tanh(x_torch_2)
    assert x_3.data == pytest.approx(x_torch_3.data.item(), 1e-3)
    print(f'x: {x_3.data}, x_torch: {x_torch_3.data.item()}')

    x_4 = x_3.exp()
    x_torch_4 = torch.exp(x_torch_3)
    assert x_4.data == pytest.approx(x_torch_4.data.item(), 1e-3)
    print(f'x: {x_4.data}, x_torch: {x_torch_4.data.item()}')

    x_4.backward()
    x_torch_4.backward()
    assert x_1.grad != 0
    assert x_1.grad == pytest.approx(x_torch_1.grad.item(), 1e-3)
    print(f'x.grad: {x_1.grad}, x_torch.grad: {x_torch_1.grad.item()}')
