import torch

def hook_fn_print_grad(grad):
    print("hook_fn_print_grad: ", grad)

def hook_fn_2grad(grad):
    g = 2 * grad
    print("hook_fn_2grad: ", g)
    return g

def SimpleNet():
    x = torch.Tensor([0, 1, 2, 3]).requires_grad_()
    y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
    w = torch.Tensor([1, 2, 3, 4]).requires_grad_()
    z = x + y
    o = torch.matmul(w, z)
    return x, z, o

def general():
    x, z, o = SimpleNet()
    o.backward()
    print('x.grad is: ', x.grad)
    print('z.grad is: ', z.grad)

def print_grad():
    x, z, o = SimpleNet()
    h = z.register_hook(hook_fn_print_grad)
    o.backward()
    print('x.grad is: ', x.grad)
    print('z.grad is: ', z.grad)
    x, z, o = SimpleNet()
    h = z.register_hook(hook_fn_print_grad)
    h.remove()
    o.backward()
    print("-----after remove hook-----")
    print('x.grad is: ', x.grad)
    print('z.grad is: ', z.grad)

def print_2grad():
    x, z, o = SimpleNet()
    h = z.register_hook(hook_fn_2grad)
    o.backward()
    print('x.grad is: ', x.grad)
    print('z.grad is: ', z.grad)
    x, z, o = SimpleNet()
    h = z.register_hook(hook_fn_2grad)
    h.remove()
    o.backward()
    print("-----after remove hook-----")
    print('x.grad is: ', x.grad)
    print('z.grad is: ', z.grad)

print("=====general=====")
general()
print("=====print_grad=====")
print_grad()
print("=====print_2grad=====")
print_2grad()

'''
=====general=====
('x.grad is: ', tensor([1., 2., 3., 4.]))
('z.grad is: ', None)
=====print_grad=====
('hook_fn_print_grad: ', tensor([1., 2., 3., 4.]))
('x.grad is: ', tensor([1., 2., 3., 4.]))
('z.grad is: ', None)
-----after remove hook-----
('x.grad is: ', tensor([1., 2., 3., 4.]))
('z.grad is: ', None)
=====print_2grad=====
('hook_fn_2grad: ', tensor([2., 4., 6., 8.]))
('x.grad is: ', tensor([2., 4., 6., 8.]))
('z.grad is: ', None)
-----after remove hook-----
('x.grad is: ', tensor([1., 2., 3., 4.]))
('z.grad is: ', None)
'''
